"""REST API endpoints for Responsible AI content safety and bias detection.

Endpoints:
- POST /responsible-ai/check-resume    — check resume for bias/PII/safety
- POST /responsible-ai/check-posting   — check job posting for discrimination
- GET  /responsible-ai/bias-report     — generate bias report from hiring history
- GET  /responsible-ai/status          — content safety statistics
- POST /responsible-ai/score           — get safety score for arbitrary text
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.responsible_ai import get_safety_checker

router = APIRouter(tags=["responsible-ai"])


# ── Request / Response models ───────────────────────────────────────────────


class TextCheckBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)


class SafetyCheckResponse(BaseModel):
    check_id: str
    content_type: str
    safety_score: float
    level: str
    issues: list[dict[str, Any]]
    bias_indicators: list[str]
    pii_detected: list[str]
    recommendations: list[str]
    checked_at: float


class SafetyScoreResponse(BaseModel):
    safety_score: float
    level: str


class BiasReportResponse(BaseModel):
    report_id: str
    total_decisions: int
    flagged_decisions: int
    bias_indicators: dict[str, int]
    recommendations: list[str]
    fairness_score: float
    generated_at: float


class SafetyStatusResponse(BaseModel):
    total_checked: int
    flagged: int
    blocked: int
    warnings: int
    safe: int
    resumes_checked: int
    job_postings_checked: int
    flagged_rate: float
    false_positive_rate: float


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.post("/responsible-ai/check-resume", response_model=SafetyCheckResponse)
async def check_resume(body: TextCheckBody):
    """Check a resume for bias indicators, PII exposure, and content safety."""
    checker = get_safety_checker()
    result = checker.check_resume(body.text)
    return SafetyCheckResponse(
        check_id=result.check_id,
        content_type=result.content_type,
        safety_score=result.safety_score,
        level=result.level.value,
        issues=result.issues,
        bias_indicators=result.bias_indicators,
        pii_detected=result.pii_detected,
        recommendations=result.recommendations,
        checked_at=result.checked_at,
    )


@router.post("/responsible-ai/check-posting", response_model=SafetyCheckResponse)
async def check_job_posting(body: TextCheckBody):
    """Check a job posting for discriminatory language and bias."""
    checker = get_safety_checker()
    result = checker.check_job_posting(body.text)
    return SafetyCheckResponse(
        check_id=result.check_id,
        content_type=result.content_type,
        safety_score=result.safety_score,
        level=result.level.value,
        issues=result.issues,
        bias_indicators=result.bias_indicators,
        pii_detected=result.pii_detected,
        recommendations=result.recommendations,
        checked_at=result.checked_at,
    )


@router.post("/responsible-ai/score", response_model=SafetyScoreResponse)
async def safety_score(body: TextCheckBody):
    """Get a safety score (0-1) for arbitrary text content."""
    checker = get_safety_checker()
    score = checker.get_safety_score(body.text)
    if score >= 0.8:
        level = "safe"
    elif score >= 0.5:
        level = "warning"
    elif score >= 0.2:
        level = "flagged"
    else:
        level = "blocked"
    return SafetyScoreResponse(safety_score=score, level=level)


@router.get("/responsible-ai/bias-report", response_model=BiasReportResponse)
async def bias_report():
    """Generate a bias report from recent hiring decisions.

    Analyzes task history for potential bias patterns.
    """
    from src.storage import get_storage
    checker = get_safety_checker()
    storage = get_storage()

    # Get recent tasks as proxy for hiring decisions
    tasks = storage.list_tasks()
    decisions = [
        {"description": t["description"], "status": t["status"]}
        for t in tasks
    ]

    report = checker.generate_bias_report(decisions)
    return BiasReportResponse(
        report_id=report.report_id,
        total_decisions=report.total_decisions,
        flagged_decisions=report.flagged_decisions,
        bias_indicators=report.bias_indicators,
        recommendations=report.recommendations,
        fairness_score=report.fairness_score,
        generated_at=report.generated_at,
    )


@router.get("/responsible-ai/status", response_model=SafetyStatusResponse)
async def safety_status():
    """Get content safety statistics.

    Shows total items checked, flagged items, false positive rate,
    and bias indicators detected.
    """
    checker = get_safety_checker()
    stats = checker.get_stats()
    return SafetyStatusResponse(**stats)
