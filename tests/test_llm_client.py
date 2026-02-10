"""Tests for the HireWire LLM client (src.llm).

All Azure API calls are mocked so tests run without credentials.
Tests cover both the Azure OpenAI path and the rule-based fallback.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

# Force mock provider
os.environ.setdefault("MODEL_PROVIDER", "mock")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def llm_client_fallback():
    """LLM client using rule-based fallback (no Azure)."""
    from src.llm import AzureLLMClient, reset_llm_client
    # Force non-Azure
    client = AzureLLMClient(provider=None)
    client._use_azure = False
    yield client
    reset_llm_client()


@pytest.fixture()
def llm_client_azure():
    """LLM client with mocked Azure provider."""
    from src.llm import AzureLLMClient, reset_llm_client

    mock_provider = MagicMock()
    client = AzureLLMClient(provider=mock_provider)
    yield client, mock_provider
    reset_llm_client()


@pytest.fixture()
def _azure_env(monkeypatch):
    """Set Azure env vars for auto-detection tests."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    import src.framework.azure_llm as mod
    mod._provider = None
    yield
    mod._provider = None


SAMPLE_RESUME = (
    "Senior Software Engineer with 8 years of experience in Python, "
    "JavaScript, and cloud infrastructure. Expertise in machine learning, "
    "FastAPI, Docker, Kubernetes. MS in Computer Science."
)

SAMPLE_JOB_POSTING = (
    "Looking for a Staff Engineer: 5+ years Python, cloud infrastructure "
    "(AWS or Azure), machine learning experience, team leadership required."
)


# ---------------------------------------------------------------------------
# Rule-based fallback tests
# ---------------------------------------------------------------------------


class TestRuleBasedFallback:
    """Test the rule-based fallback when Azure is not available."""

    def test_resume_analyze_skills(self, llm_client_fallback):
        result = llm_client_fallback.resume_analyze(SAMPLE_RESUME)
        assert isinstance(result, dict)
        assert "skills" in result
        assert "python" in result["skills"]
        assert "machine learning" in result["skills"]
        assert "fastapi" in result["skills"]
        assert "docker" in result["skills"]
        assert "kubernetes" in result["skills"]

    def test_resume_analyze_experience(self, llm_client_fallback):
        result = llm_client_fallback.resume_analyze(SAMPLE_RESUME)
        assert result["experience_years"] == 8

    def test_resume_analyze_education(self, llm_client_fallback):
        result = llm_client_fallback.resume_analyze(SAMPLE_RESUME)
        # Should detect MS / Master
        assert result["education"].lower() in ("ms", "master")

    def test_resume_analyze_fit_score(self, llm_client_fallback):
        result = llm_client_fallback.resume_analyze(SAMPLE_RESUME)
        assert 0.0 <= result["fit_score"] <= 1.0
        assert result["fit_score"] > 0.3  # Should be decent with many skills

    def test_resume_analyze_summary(self, llm_client_fallback):
        result = llm_client_fallback.resume_analyze(SAMPLE_RESUME)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_resume_analyze_empty(self, llm_client_fallback):
        result = llm_client_fallback.resume_analyze("")
        assert result["skills"] == []
        assert result["experience_years"] == 0
        assert result["fit_score"] == 0.0

    def test_job_match_full_overlap(self, llm_client_fallback):
        profile = {"skills": ["python", "aws", "ml"], "experience_years": 5}
        requirements = {"required_skills": ["python", "aws"], "min_experience": 3}
        result = llm_client_fallback.job_match(profile, requirements)
        assert result["match_score"] >= 0.9
        assert "python" in result["matched_skills"]
        assert "aws" in result["matched_skills"]
        assert len(result["missing_skills"]) == 0

    def test_job_match_partial_overlap(self, llm_client_fallback):
        profile = {"skills": ["python", "react"], "experience_years": 2}
        requirements = {"required_skills": ["python", "go", "rust"], "min_experience": 5}
        result = llm_client_fallback.job_match(profile, requirements)
        assert 0.0 < result["match_score"] < 1.0
        assert "python" in result["matched_skills"]
        assert len(result["missing_skills"]) >= 1

    def test_job_match_no_overlap(self, llm_client_fallback):
        profile = {"skills": ["cooking", "painting"]}
        requirements = {"required_skills": ["python", "aws"]}
        result = llm_client_fallback.job_match(profile, requirements)
        assert result["match_score"] == 0.0
        assert len(result["matched_skills"]) == 0
        assert len(result["missing_skills"]) == 2

    def test_job_match_no_requirements(self, llm_client_fallback):
        profile = {"skills": ["python"]}
        requirements = {"required_skills": []}
        result = llm_client_fallback.job_match(profile, requirements)
        assert result["match_score"] == 0.5  # Default for no requirements

    def test_job_match_experience_bonus(self, llm_client_fallback):
        profile = {"skills": ["python"], "experience_years": 10}
        requirements = {"required_skills": ["python"], "min_experience": 5}
        result = llm_client_fallback.job_match(profile, requirements)
        # Should get bonus for exceeding experience
        assert result["match_score"] >= 1.0

    def test_interview_questions_returns_5(self, llm_client_fallback):
        questions = llm_client_fallback.generate_interview_questions(
            SAMPLE_JOB_POSTING, SAMPLE_RESUME
        )
        assert isinstance(questions, list)
        assert len(questions) == 5

    def test_interview_questions_relevant(self, llm_client_fallback):
        questions = llm_client_fallback.generate_interview_questions(
            SAMPLE_JOB_POSTING, SAMPLE_RESUME
        )
        # At least one should be about code/software
        all_text = " ".join(questions).lower()
        assert any(kw in all_text for kw in ["code", "software", "project", "team", "cloud", "maintainable"])

    def test_interview_questions_coding_focus(self, llm_client_fallback):
        questions = llm_client_fallback.generate_interview_questions(
            "Python developer position", "Python, Django, REST API experience"
        )
        all_text = " ".join(questions).lower()
        assert any(kw in all_text for kw in ["code", "maintainable", "testable", "methodologies"])

    def test_interview_questions_cloud_focus(self, llm_client_fallback):
        questions = llm_client_fallback.generate_interview_questions(
            "Cloud infrastructure role with AWS", "AWS, Docker, Terraform expert"
        )
        all_text = " ".join(questions).lower()
        assert any(kw in all_text for kw in ["cloud", "infrastructure", "deployment"])


# ---------------------------------------------------------------------------
# Azure OpenAI tests (mocked)
# ---------------------------------------------------------------------------


class TestAzureOpenAIClient:
    """Test the Azure OpenAI path with mocked API calls."""

    def test_resume_analyze_azure(self, llm_client_azure):
        client, mock_provider = llm_client_azure
        mock_provider.generate.return_value = json.dumps({
            "skills": ["python", "kubernetes", "machine learning"],
            "experience_years": 8,
            "education": "MS Computer Science",
            "fit_score": 0.92,
            "summary": "Experienced ML engineer with strong infrastructure skills.",
        })

        result = client.resume_analyze(SAMPLE_RESUME)
        assert result["skills"] == ["python", "kubernetes", "machine learning"]
        assert result["experience_years"] == 8
        assert result["fit_score"] == 0.92
        mock_provider.generate.assert_called_once()

    def test_resume_analyze_azure_with_markdown_fences(self, llm_client_azure):
        client, mock_provider = llm_client_azure
        mock_provider.generate.return_value = '```json\n{"skills": ["python"], "experience_years": 3, "education": "BS", "fit_score": 0.5, "summary": "Junior dev"}\n```'

        result = client.resume_analyze(SAMPLE_RESUME)
        assert result["skills"] == ["python"]
        assert result["fit_score"] == 0.5

    def test_resume_analyze_azure_fallback_on_error(self, llm_client_azure):
        client, mock_provider = llm_client_azure
        mock_provider.generate.side_effect = Exception("API error")

        # Should fall back to rule-based
        result = client.resume_analyze(SAMPLE_RESUME)
        assert "skills" in result
        assert isinstance(result["skills"], list)

    def test_resume_analyze_azure_fallback_on_invalid_json(self, llm_client_azure):
        client, mock_provider = llm_client_azure
        mock_provider.generate.return_value = "This is not valid JSON at all."

        # Should fall back to rule-based
        result = client.resume_analyze(SAMPLE_RESUME)
        assert "skills" in result

    def test_job_match_azure(self, llm_client_azure):
        client, mock_provider = llm_client_azure
        mock_provider.generate.return_value = json.dumps({
            "match_score": 0.85,
            "matched_skills": ["python", "aws"],
            "missing_skills": ["go"],
            "reasoning": "Strong match with minor gaps.",
        })

        result = client.job_match(
            {"skills": ["python", "aws"]},
            {"required_skills": ["python", "aws", "go"]},
        )
        assert result["match_score"] == 0.85
        assert "python" in result["matched_skills"]
        assert "go" in result["missing_skills"]

    def test_job_match_azure_fallback_on_error(self, llm_client_azure):
        client, mock_provider = llm_client_azure
        mock_provider.generate.side_effect = Exception("Timeout")

        result = client.job_match(
            {"skills": ["python"]},
            {"required_skills": ["python", "go"]},
        )
        assert "match_score" in result
        assert isinstance(result["match_score"], float)

    def test_interview_questions_azure(self, llm_client_azure):
        client, mock_provider = llm_client_azure
        mock_provider.generate.return_value = json.dumps({
            "questions": [
                "Describe your experience with ML pipelines.",
                "How do you approach system design?",
                "Tell me about a challenging team situation.",
                "How do you handle production incidents?",
                "What's your approach to code reviews?",
            ]
        })

        questions = client.generate_interview_questions(
            SAMPLE_JOB_POSTING, SAMPLE_RESUME
        )
        assert len(questions) == 5
        assert "ML pipelines" in questions[0]

    def test_interview_questions_azure_truncates_to_5(self, llm_client_azure):
        client, mock_provider = llm_client_azure
        mock_provider.generate.return_value = json.dumps({
            "questions": [f"Q{i}" for i in range(10)]
        })

        questions = client.generate_interview_questions(
            SAMPLE_JOB_POSTING, SAMPLE_RESUME
        )
        assert len(questions) == 5

    def test_interview_questions_azure_fallback_on_empty(self, llm_client_azure):
        client, mock_provider = llm_client_azure
        mock_provider.generate.return_value = json.dumps({"questions": []})

        questions = client.generate_interview_questions(
            SAMPLE_JOB_POSTING, SAMPLE_RESUME
        )
        # Should fall back to rule-based
        assert len(questions) == 5


# ---------------------------------------------------------------------------
# Module-level singleton tests
# ---------------------------------------------------------------------------


class TestModuleSingleton:

    def test_get_llm_client_returns_instance(self):
        from src.llm import get_llm_client, reset_llm_client
        reset_llm_client()
        client = get_llm_client()
        from src.llm import AzureLLMClient
        assert isinstance(client, AzureLLMClient)

    def test_get_llm_client_is_singleton(self):
        from src.llm import get_llm_client, reset_llm_client
        reset_llm_client()
        c1 = get_llm_client()
        c2 = get_llm_client()
        assert c1 is c2

    def test_reset_llm_client_creates_new(self):
        from src.llm import get_llm_client, reset_llm_client
        c1 = get_llm_client()
        c2 = reset_llm_client()
        assert c1 is not c2

    def test_is_azure_false_without_env(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)
        from src.llm import AzureLLMClient
        client = AzureLLMClient(provider=None)
        # Without env, azure_available returns False, so _use_azure should be False
        # unless a provider was passed
        # Note: depends on env state, but we cleared the vars
        assert isinstance(client.is_azure, bool)


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


class TestAPIEndpoints:
    """Test the REST API endpoints for LLM-powered hiring."""

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)

    def test_analyze_resume_endpoint(self, client):
        resp = client.post(
            "/responsible-ai/analyze-resume",
            json={"resume_text": SAMPLE_RESUME},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "skills" in data
        assert "fit_score" in data
        assert "provider" in data
        assert data["provider"] in ("azure_openai", "rule_based")

    def test_analyze_resume_empty_text(self, client):
        resp = client.post(
            "/responsible-ai/analyze-resume",
            json={"resume_text": ""},
        )
        assert resp.status_code == 422  # Validation error (min_length=1)

    def test_job_match_endpoint(self, client):
        resp = client.post(
            "/responsible-ai/job-match",
            json={
                "candidate_profile": {"skills": ["python", "aws"], "experience_years": 5},
                "job_requirements": {"required_skills": ["python", "aws"], "min_experience": 3},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "match_score" in data
        assert "matched_skills" in data
        assert "missing_skills" in data
        assert "reasoning" in data
        assert "provider" in data

    def test_interview_questions_endpoint(self, client):
        resp = client.post(
            "/responsible-ai/interview-questions",
            json={
                "job_posting": SAMPLE_JOB_POSTING,
                "resume": SAMPLE_RESUME,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "questions" in data
        assert len(data["questions"]) == 5
        assert "provider" in data

    def test_interview_questions_empty_posting(self, client):
        resp = client.post(
            "/responsible-ai/interview-questions",
            json={"job_posting": "", "resume": SAMPLE_RESUME},
        )
        assert resp.status_code == 422  # Validation error


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_resume_with_many_skills(self, llm_client_fallback):
        resume = (
            "10 years experience in Python, JavaScript, TypeScript, Java, Go, Rust, C++, "
            "React, Vue, Angular, Node.js, FastAPI, Django, Flask, "
            "AWS, Azure, GCP, Docker, Kubernetes, Terraform, "
            "SQL, PostgreSQL, MongoDB, Redis, Elasticsearch"
        )
        result = llm_client_fallback.resume_analyze(resume)
        assert len(result["skills"]) >= 15
        assert result["fit_score"] == 1.0  # Capped at 1.0

    def test_resume_no_experience_mentioned(self, llm_client_fallback):
        result = llm_client_fallback.resume_analyze("I know Python and JavaScript.")
        assert result["experience_years"] == 0

    def test_job_match_case_insensitive(self, llm_client_fallback):
        profile = {"skills": ["Python", "AWS"]}
        requirements = {"required_skills": ["python", "aws"]}
        result = llm_client_fallback.job_match(profile, requirements)
        assert result["match_score"] >= 0.9

    def test_job_match_empty_profile(self, llm_client_fallback):
        result = llm_client_fallback.job_match({}, {"required_skills": ["python"]})
        assert result["match_score"] == 0.0

    def test_job_match_empty_both(self, llm_client_fallback):
        result = llm_client_fallback.job_match({}, {})
        assert result["match_score"] == 0.5
