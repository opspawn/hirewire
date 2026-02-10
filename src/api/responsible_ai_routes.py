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
from src.llm import get_llm_client

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


class ResumeAnalysisBody(BaseModel):
    resume_text: str = Field(..., min_length=1, max_length=50000)


class ResumeAnalysisResponse(BaseModel):
    skills: list[str]
    experience_years: int
    education: str
    fit_score: float
    summary: str
    provider: str


class JobMatchBody(BaseModel):
    candidate_profile: dict[str, Any]
    job_requirements: dict[str, Any]


class JobMatchResponse(BaseModel):
    match_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    reasoning: str
    provider: str


class InterviewQuestionsBody(BaseModel):
    job_posting: str = Field(..., min_length=1, max_length=20000)
    resume: str = Field(..., min_length=1, max_length=50000)


class InterviewQuestionsResponse(BaseModel):
    questions: list[str]
    provider: str


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


# ── LLM-powered hiring analysis endpoints ─────────────────────────────────


@router.post("/responsible-ai/analyze-resume", response_model=ResumeAnalysisResponse)
async def analyze_resume(body: ResumeAnalysisBody):
    """Analyze a resume using Azure OpenAI GPT-4o.

    Extracts skills, experience, education, and provides a fit score.
    Falls back to rule-based analysis when Azure credentials are not set.
    """
    llm = get_llm_client()
    result = llm.resume_analyze(body.resume_text)
    return ResumeAnalysisResponse(
        skills=result.get("skills", []),
        experience_years=result.get("experience_years", 0),
        education=result.get("education", "unknown"),
        fit_score=result.get("fit_score", 0.0),
        summary=result.get("summary", ""),
        provider="azure_openai" if llm.is_azure else "rule_based",
    )


@router.post("/responsible-ai/job-match", response_model=JobMatchResponse)
async def job_match(body: JobMatchBody):
    """Match a candidate profile against job requirements using Azure OpenAI.

    Returns match score, matched/missing skills, and reasoning.
    Falls back to deterministic skill overlap when Azure credentials are not set.
    """
    llm = get_llm_client()
    result = llm.job_match(body.candidate_profile, body.job_requirements)
    return JobMatchResponse(
        match_score=result.get("match_score", 0.0),
        matched_skills=result.get("matched_skills", []),
        missing_skills=result.get("missing_skills", []),
        reasoning=result.get("reasoning", ""),
        provider="azure_openai" if llm.is_azure else "rule_based",
    )


@router.post("/responsible-ai/interview-questions", response_model=InterviewQuestionsResponse)
async def interview_questions(body: InterviewQuestionsBody):
    """Generate tailored interview questions using Azure OpenAI.

    Creates 5 questions based on the job posting and candidate resume.
    Falls back to keyword-based question generation when Azure is not set.
    """
    llm = get_llm_client()
    questions = llm.generate_interview_questions(body.job_posting, body.resume)
    return InterviewQuestionsResponse(
        questions=questions,
        provider="azure_openai" if llm.is_azure else "rule_based",
    )
