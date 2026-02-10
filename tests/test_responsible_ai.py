"""Tests for the Responsible AI content safety module.

Tests cover:
- Resume content safety checking
- Job posting validation
- Bias detection (gender, age, ethnicity, disability, religion)
- PII detection (SSN, email, phone, credit card)
- Safety scoring
- Bias report generation
- Discriminatory phrase detection
- Statistics tracking
- Edge cases
"""

import pytest

from src.responsible_ai import (
    ContentSafetyChecker,
    SafetyCheckResult,
    SafetyLevel,
    BiasCategory,
    BiasReport,
    get_safety_checker,
    reset_safety_checker,
)


@pytest.fixture
def checker():
    """Create a fresh ContentSafetyChecker for each test."""
    return ContentSafetyChecker()


# ── Resume checking ────────────────────────────────────────────────────────


class TestCheckResume:
    def test_safe_resume(self, checker):
        result = checker.check_resume(
            "Experienced software engineer with 5 years of Python experience. "
            "Built scalable microservices and REST APIs."
        )
        assert result.safety_score >= 0.8
        assert result.level == SafetyLevel.SAFE
        assert result.content_type == "resume"

    def test_resume_with_pii_ssn(self, checker):
        result = checker.check_resume(
            "John Doe, SSN: 123-45-6789. Software Engineer."
        )
        assert "SSN" in result.pii_detected
        assert result.safety_score < 1.0
        assert len(result.recommendations) > 0

    def test_resume_with_email(self, checker):
        result = checker.check_resume(
            "Jane Smith, contact: jane@example.com. Data Scientist."
        )
        assert "email" in result.pii_detected

    def test_resume_with_phone(self, checker):
        result = checker.check_resume(
            "Contact: 555-123-4567. Available for interviews."
        )
        assert "phone" in result.pii_detected

    def test_resume_with_gender_indicators(self, checker):
        result = checker.check_resume(
            "He is a skilled engineer with leadership experience."
        )
        assert len(result.bias_indicators) > 0

    def test_resume_with_multiple_pii(self, checker):
        result = checker.check_resume(
            "SSN: 123-45-6789, Email: test@test.com, Phone: 555-111-2222"
        )
        assert len(result.pii_detected) >= 3
        assert result.safety_score < 0.7

    def test_resume_check_id(self, checker):
        result = checker.check_resume("Simple resume text")
        assert result.check_id.startswith("chk_")

    def test_resume_inappropriate_content(self, checker):
        result = checker.check_resume(
            "This candidate has a history of hate speech and violent behavior"
        )
        assert result.safety_score < 1.0
        assert any(
            i["type"] == "inappropriate_content" for i in result.issues
        )


# ── Job posting checking ──────────────────────────────────────────────────


class TestCheckJobPosting:
    def test_safe_job_posting(self, checker):
        result = checker.check_job_posting(
            "Software Engineer needed. 3+ years experience with Python. "
            "Remote-friendly team. Competitive salary."
        )
        assert result.safety_score >= 0.8
        assert result.level == SafetyLevel.SAFE
        assert result.content_type == "job_posting"

    def test_discriminatory_age_posting(self, checker):
        result = checker.check_job_posting(
            "Looking for young and energetic candidates to join our team."
        )
        assert result.safety_score < 1.0
        assert any(
            i["type"] == "discriminatory_language" for i in result.issues
        )

    def test_discriminatory_graduate_posting(self, checker):
        result = checker.check_job_posting(
            "Recent graduate only. Must be fresh out of college."
        )
        assert result.safety_score < 1.0

    def test_discriminatory_native_speaker(self, checker):
        result = checker.check_job_posting(
            "Must be a native english speaker with no disabilities."
        )
        assert result.safety_score < 0.8

    def test_gender_biased_posting(self, checker):
        result = checker.check_job_posting(
            "Looking for a salesman to join our brotherhood of sales professionals."
        )
        assert len(result.bias_indicators) > 0
        assert any(
            i.get("category") == "gender" for i in result.issues
        )

    def test_posting_with_pii(self, checker):
        result = checker.check_job_posting(
            "Contact hiring manager at 555-123-4567 for details."
        )
        assert "phone" in result.pii_detected

    def test_culture_fit_posting(self, checker):
        result = checker.check_job_posting(
            "We're looking for someone who is a great culture fit."
        )
        assert result.safety_score < 1.0

    def test_multiple_discriminatory_phrases(self, checker):
        result = checker.check_job_posting(
            "Young and energetic digital native with must be under 30. "
            "Recent graduate only. Able-bodied candidates preferred."
        )
        assert result.safety_score < 0.5
        disc_issues = [
            i for i in result.issues
            if i["type"] == "discriminatory_language"
        ]
        assert len(disc_issues) >= 2


# ── Safety scoring ─────────────────────────────────────────────────────────


class TestGetSafetyScore:
    def test_safe_text(self, checker):
        score = checker.get_safety_score("A well-written professional document.")
        assert score >= 0.8

    def test_text_with_pii(self, checker):
        score = checker.get_safety_score("SSN: 123-45-6789")
        assert score < 1.0

    def test_text_with_discrimination(self, checker):
        score = checker.get_safety_score("young and energetic candidates only")
        assert score < 1.0

    def test_score_range(self, checker):
        score = checker.get_safety_score("test")
        assert 0.0 <= score <= 1.0

    def test_clean_professional_text(self, checker):
        score = checker.get_safety_score("Clean document about software engineering.")
        assert score >= 0.9


# ── Bias detection ─────────────────────────────────────────────────────────


class TestBiasDetection:
    def test_gender_detection(self, checker):
        result = checker.check_resume("He is a skilled chairman with leadership.")
        categories = [
            i.get("category")
            for i in result.issues
            if i["type"] == "bias_indicator"
        ]
        assert "gender" in categories

    def test_age_detection(self, checker):
        result = checker.check_job_posting(
            "Looking for young and junior developers."
        )
        bias = [
            i for i in result.issues
            if i.get("category") == "age"
        ]
        assert len(bias) > 0

    def test_ethnicity_detection(self, checker):
        result = checker.check_resume(
            "Member of the minority engineers association. Active in ethnic community."
        )
        has_ethnicity = any(
            i.get("category") == "ethnicity"
            for i in result.issues
            if i["type"] == "bias_indicator"
        )
        assert has_ethnicity

    def test_disability_detection(self, checker):
        result = checker.check_resume(
            "The disabled candidate was rejected. Must be able-bodied."
        )
        has_disability = any(
            i.get("category") == "disability"
            for i in result.issues
            if i["type"] == "bias_indicator"
        )
        assert has_disability

    def test_religion_detection(self, checker):
        result = checker.check_resume(
            "Active member of local church. Volunteers at mosque."
        )
        has_religion = any(
            i.get("category") == "religion"
            for i in result.issues
            if i["type"] == "bias_indicator"
        )
        assert has_religion

    def test_no_bias_in_neutral_text(self, checker):
        result = checker.check_resume(
            "Experienced software engineer skilled in Python, Docker, and Kubernetes."
        )
        bias_issues = [
            i for i in result.issues if i["type"] == "bias_indicator"
        ]
        assert len(bias_issues) == 0


# ── PII detection ─────────────────────────────────────────────────────────


class TestPIIDetection:
    def test_ssn_with_dashes(self, checker):
        result = checker.check_resume("SSN: 123-45-6789")
        assert "SSN" in result.pii_detected

    def test_email_detection(self, checker):
        result = checker.check_resume("Email: user@example.com")
        assert "email" in result.pii_detected

    def test_phone_detection(self, checker):
        result = checker.check_resume("Phone: 555-123-4567")
        assert "phone" in result.pii_detected

    def test_credit_card_detection(self, checker):
        result = checker.check_resume("Card: 4111 1111 1111 1111")
        assert "credit_card" in result.pii_detected

    def test_no_pii_in_clean_text(self, checker):
        result = checker.check_resume(
            "Professional software developer with expertise in cloud computing."
        )
        assert len(result.pii_detected) == 0


# ── Bias report ────────────────────────────────────────────────────────────


class TestBiasReport:
    def test_empty_decisions(self, checker):
        report = checker.generate_bias_report([])
        assert report.total_decisions == 0
        assert report.fairness_score == 1.0

    def test_clean_decisions(self, checker):
        decisions = [
            {"description": "Build a REST API"},
            {"description": "Deploy to cloud"},
            {"description": "Run integration tests"},
        ]
        report = checker.generate_bias_report(decisions)
        assert report.total_decisions == 3
        assert report.flagged_decisions == 0
        assert report.fairness_score == 1.0

    def test_biased_decisions(self, checker):
        decisions = [
            {"description": "Hire a young male engineer"},
            {"description": "Find female designer"},
            {"description": "Build a REST API"},
        ]
        report = checker.generate_bias_report(decisions)
        assert report.total_decisions == 3
        assert report.flagged_decisions >= 2
        assert report.fairness_score < 1.0
        assert len(report.recommendations) > 0

    def test_report_has_id(self, checker):
        report = checker.generate_bias_report([])
        assert report.report_id.startswith("bias_")

    def test_report_to_dict(self, checker):
        report = checker.generate_bias_report([])
        d = report.to_dict()
        assert "report_id" in d
        assert "fairness_score" in d

    def test_gender_bias_recommendation(self, checker):
        decisions = [
            {"description": "He should handle the frontend"},
            {"description": "She will manage the backend"},
        ]
        report = checker.generate_bias_report(decisions)
        has_gender_rec = any(
            "gender" in r.lower() for r in report.recommendations
        )
        assert has_gender_rec


# ── Statistics ─────────────────────────────────────────────────────────────


class TestStatistics:
    def test_initial_stats(self, checker):
        stats = checker.get_stats()
        assert stats["total_checked"] == 0

    def test_stats_after_checks(self, checker):
        checker.check_resume("A safe resume")
        checker.check_job_posting("A safe posting")
        stats = checker.get_stats()
        assert stats["total_checked"] == 2
        assert stats["resumes_checked"] == 1
        assert stats["job_postings_checked"] == 1

    def test_flagged_rate(self, checker):
        checker.check_resume("Safe resume text")
        checker.check_job_posting(
            "Young and energetic digital native recent graduate only "
            "culture fit able-bodied no disabilities"
        )
        stats = checker.get_stats()
        assert stats["flagged_rate"] >= 0.0


# ── SafetyCheckResult ─────────────────────────────────────────────────────


class TestSafetyCheckResult:
    def test_to_dict(self):
        result = SafetyCheckResult(
            content_type="resume",
            safety_score=0.9,
            level=SafetyLevel.SAFE,
        )
        d = result.to_dict()
        assert d["level"] == "safe"
        assert d["safety_score"] == 0.9

    def test_check_id_format(self):
        result = SafetyCheckResult()
        assert result.check_id.startswith("chk_")


# ── Singleton ──────────────────────────────────────────────────────────────


class TestSingleton:
    def test_get_safety_checker(self):
        checker = get_safety_checker()
        assert isinstance(checker, ContentSafetyChecker)

    def test_reset_safety_checker(self):
        old = get_safety_checker()
        new = reset_safety_checker()
        assert old is not new


# ── Edge cases ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_very_long_text(self, checker):
        text = "Python developer with skills. " * 1000
        result = checker.check_resume(text)
        assert result.safety_score >= 0.0

    def test_unicode_text(self, checker):
        result = checker.check_resume(
            "Développeur logiciel avec 5 ans d'expérience en Python."
        )
        assert result.safety_score >= 0.0

    def test_numbers_only(self, checker):
        result = checker.check_resume("12345 67890 11111")
        assert result.content_type == "resume"

    def test_clear(self, checker):
        checker.check_resume("test")
        checker.clear()
        stats = checker.get_stats()
        assert stats["total_checked"] == 0

    def test_get_recent_checks(self, checker):
        checker.check_resume("Resume 1")
        checker.check_resume("Resume 2")
        recent = checker.get_recent_checks(limit=1)
        assert len(recent) == 1

    def test_safety_level_values(self):
        assert SafetyLevel.SAFE.value == "safe"
        assert SafetyLevel.WARNING.value == "warning"
        assert SafetyLevel.FLAGGED.value == "flagged"
        assert SafetyLevel.BLOCKED.value == "blocked"

    def test_bias_category_values(self):
        assert BiasCategory.GENDER.value == "gender"
        assert BiasCategory.AGE.value == "age"
        assert BiasCategory.NONE.value == "none"
