"""Responsible AI module for HireWire.

Provides content safety checks, bias detection, and fairness monitoring
for the hiring pipeline. Ensures resumes and job postings are screened
for discriminatory language, PII exposure, and inappropriate content.

Features:
- Resume content safety checking
- Job posting validation
- Bias detection and reporting
- Safety scoring (0-1)
- PII detection
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class SafetyLevel(str, Enum):
    """Safety assessment level."""

    SAFE = "safe"
    WARNING = "warning"
    FLAGGED = "flagged"
    BLOCKED = "blocked"


class BiasCategory(str, Enum):
    """Categories of potential bias."""

    GENDER = "gender"
    AGE = "age"
    ETHNICITY = "ethnicity"
    DISABILITY = "disability"
    RELIGION = "religion"
    NONE = "none"


@dataclass
class SafetyCheckResult:
    """Result of a content safety check."""

    check_id: str = field(default_factory=lambda: f"chk_{uuid.uuid4().hex[:12]}")
    content_type: str = ""  # "resume", "job_posting", "general"
    safety_score: float = 1.0  # 0.0 (unsafe) to 1.0 (safe)
    level: SafetyLevel = SafetyLevel.SAFE
    issues: list[dict[str, Any]] = field(default_factory=list)
    bias_indicators: list[str] = field(default_factory=list)
    pii_detected: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    checked_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        return d


@dataclass
class BiasReport:
    """Bias analysis report for hiring decisions."""

    report_id: str = field(default_factory=lambda: f"bias_{uuid.uuid4().hex[:12]}")
    total_decisions: int = 0
    flagged_decisions: int = 0
    bias_indicators: dict[str, int] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    fairness_score: float = 1.0  # 0.0 (unfair) to 1.0 (fair)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Bias / PII patterns ─────────────────────────────────────────────────────

_GENDER_TERMS = {
    "he", "she", "him", "her", "his", "hers", "male", "female",
    "man", "woman", "boy", "girl", "gentleman", "lady",
    "husband", "wife", "father", "mother", "son", "daughter",
    "brotherhood", "sisterhood", "manpower", "mankind",
    "chairman", "chairwoman", "salesman", "saleswoman",
    "fireman", "policeman", "policewoman", "congressman",
    "waitress", "stewardess", "actress",
}

_AGE_TERMS = {
    "young", "old", "elderly", "senior", "junior", "mature",
    "youthful", "aged", "millennial", "boomer", "gen-z",
    "recent graduate", "fresh graduate", "experienced veteran",
    "digital native",
}

_ETHNICITY_TERMS = {
    "race", "racial", "ethnic", "ethnicity", "minority",
    "caucasian", "african", "asian", "hispanic", "latino",
    "latina", "native", "indigenous", "tribal", "color",
    "white", "black",  # contextual — only flagged as indicators
}

_DISABILITY_TERMS = {
    "disabled", "handicapped", "impaired", "wheelchair",
    "blind", "deaf", "mute", "crippled", "lame",
    "able-bodied", "normal",  # when contrasted
}

_RELIGION_TERMS = {
    "christian", "muslim", "jewish", "hindu", "buddhist",
    "atheist", "agnostic", "religious", "church", "mosque",
    "temple", "synagogue",
}

_DISCRIMINATORY_JOB_PHRASES = [
    "young and energetic",
    "recent graduate only",
    "must be under",
    "must be over",
    "native english speaker",
    "no disabilities",
    "able-bodied",
    "culture fit",
    "digital native",
    "fresh out of college",
    "mature candidates only",
    "young professionals",
]

_PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
    (r"\b\d{9}\b", "SSN (no dashes)"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone"),
    (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "credit_card"),
    (r"\b(?:0[1-9]|1[0-2])/\d{2,4}\b", "date_of_birth"),
]


class ContentSafetyChecker:
    """Content safety checker for the hiring pipeline.

    Provides bias detection, PII scanning, and safety scoring
    for resumes, job postings, and general text content.
    """

    def __init__(self) -> None:
        self._checks: list[SafetyCheckResult] = []
        self._stats = {
            "total_checked": 0,
            "flagged": 0,
            "blocked": 0,
            "warnings": 0,
            "safe": 0,
            "resumes_checked": 0,
            "job_postings_checked": 0,
        }

    def _detect_bias_indicators(
        self, text: str
    ) -> list[tuple[BiasCategory, list[str]]]:
        """Detect potential bias indicators in text."""
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+(?:-\w+)*\b", text_lower))
        results: list[tuple[BiasCategory, list[str]]] = []

        gender_found = words & _GENDER_TERMS
        if gender_found:
            results.append((BiasCategory.GENDER, sorted(gender_found)))

        age_found = words & _AGE_TERMS
        if age_found:
            results.append((BiasCategory.AGE, sorted(age_found)))

        ethnicity_found = words & _ETHNICITY_TERMS
        if ethnicity_found:
            results.append((BiasCategory.ETHNICITY, sorted(ethnicity_found)))

        disability_found = words & _DISABILITY_TERMS
        if disability_found:
            results.append((BiasCategory.DISABILITY, sorted(disability_found)))

        religion_found = words & _RELIGION_TERMS
        if religion_found:
            results.append((BiasCategory.RELIGION, sorted(religion_found)))

        return results

    def _detect_pii(self, text: str) -> list[str]:
        """Detect PII (Personally Identifiable Information) in text."""
        found = []
        for pattern, label in _PII_PATTERNS:
            if re.search(pattern, text):
                found.append(label)
        return found

    def _detect_discriminatory_phrases(self, text: str) -> list[str]:
        """Detect discriminatory phrases in text."""
        text_lower = text.lower()
        return [
            phrase
            for phrase in _DISCRIMINATORY_JOB_PHRASES
            if phrase in text_lower
        ]

    def check_resume(self, text: str) -> SafetyCheckResult:
        """Check a resume for bias indicators, PII exposure, and content safety.

        Returns a SafetyCheckResult with safety score and any issues found.
        """
        result = SafetyCheckResult(content_type="resume")
        issues = []
        score = 1.0

        # Check for PII
        pii = self._detect_pii(text)
        if pii:
            result.pii_detected = pii
            issues.append({
                "type": "pii_exposure",
                "severity": "warning",
                "details": f"PII detected: {', '.join(pii)}",
            })
            score -= 0.15 * len(pii)
            result.recommendations.append(
                "Remove or redact personally identifiable information before processing"
            )

        # Check for bias indicators (informational — resumes naturally contain some)
        bias = self._detect_bias_indicators(text)
        bias_labels = []
        for category, terms in bias:
            bias_labels.extend(terms)
            issues.append({
                "type": "bias_indicator",
                "severity": "info",
                "category": category.value,
                "terms": terms,
                "details": f"{category.value} terms detected: {', '.join(terms)}",
            })
        result.bias_indicators = bias_labels

        # Check for inappropriate content
        inappropriate = self._check_inappropriate_content(text)
        if inappropriate:
            issues.extend(inappropriate)
            score -= 0.3

        # Calculate final score
        result.safety_score = max(0.0, min(1.0, score))
        result.issues = issues

        # Determine level
        if result.safety_score >= 0.8:
            result.level = SafetyLevel.SAFE
        elif result.safety_score >= 0.5:
            result.level = SafetyLevel.WARNING
        elif result.safety_score >= 0.2:
            result.level = SafetyLevel.FLAGGED
        else:
            result.level = SafetyLevel.BLOCKED

        # Update stats
        self._record_check(result)
        return result

    def check_job_posting(self, text: str) -> SafetyCheckResult:
        """Check a job posting for discriminatory language and bias.

        Returns a SafetyCheckResult with safety score and any issues found.
        """
        result = SafetyCheckResult(content_type="job_posting")
        issues = []
        score = 1.0

        # Check for discriminatory phrases
        disc_phrases = self._detect_discriminatory_phrases(text)
        if disc_phrases:
            for phrase in disc_phrases:
                issues.append({
                    "type": "discriminatory_language",
                    "severity": "flagged",
                    "details": f"Discriminatory phrase: '{phrase}'",
                })
            score -= 0.2 * len(disc_phrases)
            result.recommendations.append(
                "Remove discriminatory language and use inclusive alternatives"
            )

        # Check for gender-coded language
        bias = self._detect_bias_indicators(text)
        bias_labels = []
        for category, terms in bias:
            if category in (BiasCategory.GENDER, BiasCategory.AGE):
                bias_labels.extend(terms)
                issues.append({
                    "type": "bias_indicator",
                    "severity": "warning",
                    "category": category.value,
                    "terms": terms,
                    "details": f"Potentially biased {category.value} language: {', '.join(terms)}",
                })
                score -= 0.1
                result.recommendations.append(
                    f"Consider using gender-neutral / age-neutral alternatives for: {', '.join(terms)}"
                )
            else:
                bias_labels.extend(terms)
        result.bias_indicators = bias_labels

        # Check for PII (shouldn't be in job postings)
        pii = self._detect_pii(text)
        if pii:
            result.pii_detected = pii
            issues.append({
                "type": "pii_exposure",
                "severity": "warning",
                "details": f"PII in job posting: {', '.join(pii)}",
            })
            score -= 0.15

        # Calculate final score
        result.safety_score = max(0.0, min(1.0, score))
        result.issues = issues

        if result.safety_score >= 0.8:
            result.level = SafetyLevel.SAFE
        elif result.safety_score >= 0.5:
            result.level = SafetyLevel.WARNING
        elif result.safety_score >= 0.2:
            result.level = SafetyLevel.FLAGGED
        else:
            result.level = SafetyLevel.BLOCKED

        self._record_check(result)
        return result

    def get_safety_score(self, text: str) -> float:
        """Get a safety score (0-1) for arbitrary text content.

        Higher scores indicate safer content. This is a lightweight check
        that combines bias, PII, and inappropriate content detection.
        """
        score = 1.0

        pii = self._detect_pii(text)
        score -= 0.15 * len(pii)

        bias = self._detect_bias_indicators(text)
        score -= 0.05 * len(bias)

        disc = self._detect_discriminatory_phrases(text)
        score -= 0.2 * len(disc)

        inappropriate = self._check_inappropriate_content(text)
        score -= 0.3 * len(inappropriate)

        return max(0.0, min(1.0, score))

    def generate_bias_report(
        self, hiring_decisions: list[dict[str, Any]]
    ) -> BiasReport:
        """Analyze hiring decisions for potential bias patterns.

        Args:
            hiring_decisions: List of dicts with keys like 'agent_id',
                'task_description', 'status', 'cost', etc.

        Returns:
            BiasReport with fairness analysis and recommendations.
        """
        report = BiasReport(total_decisions=len(hiring_decisions))
        indicator_counts: dict[str, int] = {}
        flagged = 0

        for decision in hiring_decisions:
            desc = decision.get("task_description", "") or decision.get("description", "")
            bias = self._detect_bias_indicators(desc)
            if bias:
                flagged += 1
                for category, terms in bias:
                    key = category.value
                    indicator_counts[key] = indicator_counts.get(key, 0) + len(terms)

        report.flagged_decisions = flagged
        report.bias_indicators = indicator_counts

        # Calculate fairness score
        if hiring_decisions:
            flagged_rate = flagged / len(hiring_decisions)
            report.fairness_score = max(0.0, 1.0 - flagged_rate)
        else:
            report.fairness_score = 1.0

        # Generate recommendations
        if indicator_counts.get("gender", 0) > 0:
            report.recommendations.append(
                "Review hiring pipeline for gender bias — consider blind resume screening"
            )
        if indicator_counts.get("age", 0) > 0:
            report.recommendations.append(
                "Age-related terms detected — ensure job requirements focus on skills, not age"
            )
        if indicator_counts.get("ethnicity", 0) > 0:
            report.recommendations.append(
                "Ethnicity indicators detected — ensure evaluation criteria are objective and skill-based"
            )
        if not indicator_counts:
            report.recommendations.append(
                "No bias indicators detected — continue monitoring for fairness"
            )

        return report

    def _check_inappropriate_content(self, text: str) -> list[dict[str, Any]]:
        """Check for inappropriate content (placeholder for production ML model)."""
        issues = []
        # Simple heuristic — in production, use Azure Content Safety API
        text_lower = text.lower()
        offensive_patterns = [
            r"\b(hate|violent|threat|harass|abuse)\b",
        ]
        for pattern in offensive_patterns:
            if re.search(pattern, text_lower):
                issues.append({
                    "type": "inappropriate_content",
                    "severity": "flagged",
                    "details": "Potentially inappropriate content detected",
                })
                break
        return issues

    def _record_check(self, result: SafetyCheckResult) -> None:
        """Record a check result in stats."""
        self._checks.append(result)
        self._stats["total_checked"] += 1

        if result.content_type == "resume":
            self._stats["resumes_checked"] += 1
        elif result.content_type == "job_posting":
            self._stats["job_postings_checked"] += 1

        if result.level == SafetyLevel.SAFE:
            self._stats["safe"] += 1
        elif result.level == SafetyLevel.WARNING:
            self._stats["warnings"] += 1
        elif result.level == SafetyLevel.FLAGGED:
            self._stats["flagged"] += 1
        elif result.level == SafetyLevel.BLOCKED:
            self._stats["blocked"] += 1

    def get_stats(self) -> dict[str, Any]:
        """Get content safety statistics."""
        total = self._stats["total_checked"]
        flagged = self._stats["flagged"] + self._stats["blocked"]
        return {
            **self._stats,
            "flagged_rate": flagged / total if total > 0 else 0.0,
            "false_positive_rate": 0.0,  # Would need human review data
        }

    def get_recent_checks(self, limit: int = 20) -> list[SafetyCheckResult]:
        """Get recent check results."""
        return sorted(
            self._checks[-limit:],
            key=lambda c: c.checked_at,
            reverse=True,
        )

    def clear(self) -> None:
        """Clear all checks and reset stats. For testing."""
        self._checks.clear()
        self._stats = {
            "total_checked": 0,
            "flagged": 0,
            "blocked": 0,
            "warnings": 0,
            "safe": 0,
            "resumes_checked": 0,
            "job_postings_checked": 0,
        }


# Module-level singleton
_safety_checker: ContentSafetyChecker | None = None


def get_safety_checker() -> ContentSafetyChecker:
    """Get or create the global content safety checker."""
    global _safety_checker
    if _safety_checker is None:
        _safety_checker = ContentSafetyChecker()
    return _safety_checker


def reset_safety_checker() -> ContentSafetyChecker:
    """Reset the global safety checker (for testing)."""
    global _safety_checker
    _safety_checker = ContentSafetyChecker()
    return _safety_checker


__all__ = [
    "ContentSafetyChecker",
    "SafetyCheckResult",
    "SafetyLevel",
    "BiasCategory",
    "BiasReport",
    "get_safety_checker",
    "reset_safety_checker",
]
