"""Tests for Responsible AI API endpoints.

Tests cover all /responsible-ai/* endpoints via FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.responsible_ai import reset_safety_checker


@pytest.fixture(autouse=True)
def _reset_checker():
    """Reset safety checker between tests."""
    reset_safety_checker()
    yield
    reset_safety_checker()


@pytest.fixture
def client():
    return TestClient(app)


class TestCheckResumeEndpoint:
    def test_safe_resume(self, client):
        resp = client.post(
            "/responsible-ai/check-resume",
            json={"text": "Experienced Python developer with cloud expertise."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content_type"] == "resume"
        assert data["safety_score"] >= 0.8
        assert data["level"] == "safe"

    def test_resume_with_pii(self, client):
        resp = client.post(
            "/responsible-ai/check-resume",
            json={"text": "John Doe, SSN: 123-45-6789, email: john@test.com"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "SSN" in data["pii_detected"]
        assert "email" in data["pii_detected"]
        assert data["safety_score"] < 1.0

    def test_resume_with_bias(self, client):
        resp = client.post(
            "/responsible-ai/check-resume",
            json={"text": "He is an experienced chairman and leader."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["bias_indicators"]) > 0

    def test_empty_text_rejected(self, client):
        resp = client.post(
            "/responsible-ai/check-resume",
            json={"text": ""},
        )
        assert resp.status_code == 422  # Validation error


class TestCheckPostingEndpoint:
    def test_safe_posting(self, client):
        resp = client.post(
            "/responsible-ai/check-posting",
            json={"text": "Software Engineer needed. Remote-friendly team."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content_type"] == "job_posting"
        assert data["level"] == "safe"

    def test_discriminatory_posting(self, client):
        resp = client.post(
            "/responsible-ai/check-posting",
            json={"text": "Looking for young and energetic candidates who are digital native."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["safety_score"] < 1.0
        assert len(data["issues"]) > 0


class TestSafetyScoreEndpoint:
    def test_score_safe_text(self, client):
        resp = client.post(
            "/responsible-ai/score",
            json={"text": "Professional resume for a software engineer."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["safety_score"] >= 0.8
        assert data["level"] == "safe"

    def test_score_unsafe_text(self, client):
        resp = client.post(
            "/responsible-ai/score",
            json={"text": "SSN: 123-45-6789. Young and energetic only."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["safety_score"] < 1.0


class TestBiasReportEndpoint:
    def test_bias_report(self, client):
        resp = client.get("/responsible-ai/bias-report")
        assert resp.status_code == 200
        data = resp.json()
        assert "report_id" in data
        assert "fairness_score" in data
        assert "recommendations" in data
        assert data["fairness_score"] >= 0.0


class TestStatusEndpoint:
    def test_initial_status(self, client):
        resp = client.get("/responsible-ai/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_checked"] == 0

    def test_status_after_checks(self, client):
        client.post(
            "/responsible-ai/check-resume",
            json={"text": "Test resume."},
        )
        client.post(
            "/responsible-ai/check-posting",
            json={"text": "Test job posting."},
        )
        resp = client.get("/responsible-ai/status")
        data = resp.json()
        assert data["total_checked"] == 2
        assert data["resumes_checked"] == 1
        assert data["job_postings_checked"] == 1
