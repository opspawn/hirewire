"""Azure OpenAI LLM client for HireWire hiring workflows.

Provides hiring-specific LLM functions (resume analysis, job matching,
interview question generation) backed by Azure OpenAI GPT-4o.

Falls back to rule-based scoring when Azure credentials are not configured,
enabling local development and testing without API keys.

Environment variables:
    AZURE_OPENAI_ENDPOINT   — Azure OpenAI resource endpoint
    AZURE_OPENAI_KEY        — API key
    AZURE_OPENAI_DEPLOYMENT — Model deployment name (default: gpt-4o)
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.framework.azure_llm import AzureLLMProvider, azure_available, get_azure_llm


class AzureLLMClient:
    """Hiring-specific LLM client backed by Azure OpenAI.

    When Azure credentials are available, uses GPT-4o for intelligent
    resume analysis, job matching, and interview question generation.
    Falls back to deterministic rule-based scoring otherwise.
    """

    def __init__(self, provider: AzureLLMProvider | None = None) -> None:
        self._provider = provider
        self._use_azure = provider is not None or azure_available()

    @property
    def provider(self) -> AzureLLMProvider | None:
        """Lazy-load the Azure LLM provider."""
        if self._provider is None and self._use_azure:
            self._provider = get_azure_llm()
        return self._provider

    @property
    def is_azure(self) -> bool:
        """True if using real Azure OpenAI, False if rule-based fallback."""
        return self._use_azure

    # ------------------------------------------------------------------
    # Resume Analysis
    # ------------------------------------------------------------------

    def resume_analyze(self, resume_text: str) -> dict[str, Any]:
        """Analyze a resume and extract skills, experience, and fit score.

        Args:
            resume_text: Raw text content of the resume.

        Returns:
            Dict with ``skills``, ``experience_years``, ``education``,
            ``fit_score`` (0-1), and ``summary``.
        """
        if not self._use_azure:
            return self._rule_based_resume_analyze(resume_text)

        prompt = (
            "Analyze the following resume and return a JSON object with these fields:\n"
            "- skills: list of technical and soft skills\n"
            "- experience_years: estimated years of experience (integer)\n"
            "- education: highest education level\n"
            "- fit_score: overall quality score from 0.0 to 1.0\n"
            "- summary: 2-3 sentence summary of the candidate\n\n"
            "Return ONLY valid JSON, no markdown fences.\n\n"
            f"Resume:\n{resume_text}"
        )
        return self._call_json(
            prompt,
            system="You are an expert HR analyst. Analyze resumes objectively without bias.",
            fallback=lambda: self._rule_based_resume_analyze(resume_text),
        )

    # ------------------------------------------------------------------
    # Job Matching
    # ------------------------------------------------------------------

    def job_match(
        self,
        candidate_profile: dict[str, Any],
        job_requirements: dict[str, Any],
    ) -> dict[str, Any]:
        """Score how well a candidate matches job requirements.

        Args:
            candidate_profile: Dict with skills, experience, education.
            job_requirements: Dict with required_skills, min_experience, etc.

        Returns:
            Dict with ``match_score`` (0-1), ``matched_skills``,
            ``missing_skills``, and ``reasoning``.
        """
        if not self._use_azure:
            return self._rule_based_job_match(candidate_profile, job_requirements)

        prompt = (
            "Compare this candidate profile against the job requirements.\n"
            "Return a JSON object with:\n"
            "- match_score: 0.0 to 1.0 indicating fit\n"
            "- matched_skills: list of skills the candidate has that match\n"
            "- missing_skills: list of required skills the candidate lacks\n"
            "- reasoning: 2-3 sentences explaining the match assessment\n\n"
            "Return ONLY valid JSON, no markdown fences.\n\n"
            f"Candidate Profile:\n{json.dumps(candidate_profile, indent=2)}\n\n"
            f"Job Requirements:\n{json.dumps(job_requirements, indent=2)}"
        )
        return self._call_json(
            prompt,
            system="You are an expert recruiter. Evaluate candidates objectively based on skills and experience.",
            fallback=lambda: self._rule_based_job_match(candidate_profile, job_requirements),
        )

    # ------------------------------------------------------------------
    # Interview Question Generation
    # ------------------------------------------------------------------

    def generate_interview_questions(
        self,
        job_posting: str,
        resume: str,
    ) -> list[str]:
        """Generate tailored interview questions based on the job and resume.

        Args:
            job_posting: The job posting text.
            resume: The candidate's resume text.

        Returns:
            List of 5 interview questions.
        """
        if not self._use_azure:
            return self._rule_based_interview_questions(job_posting, resume)

        prompt = (
            "Generate exactly 5 tailored interview questions for this candidate "
            "based on the job posting and their resume.\n"
            "Return a JSON object with a single key \"questions\" containing a list of 5 strings.\n\n"
            "Return ONLY valid JSON, no markdown fences.\n\n"
            f"Job Posting:\n{job_posting}\n\n"
            f"Resume:\n{resume}"
        )
        result = self._call_json(
            prompt,
            system="You are an expert interviewer. Create insightful questions that assess both technical skills and cultural fit.",
            fallback=lambda: {"questions": self._rule_based_interview_questions(job_posting, resume)},
        )
        questions = result.get("questions", [])
        if isinstance(questions, list) and len(questions) >= 1:
            return questions[:5]
        return self._rule_based_interview_questions(job_posting, resume)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_json(
        self,
        prompt: str,
        system: str,
        fallback: Any,
    ) -> dict[str, Any]:
        """Call Azure OpenAI and parse JSON from the response.

        Falls back to the provided fallback function on any error.
        """
        try:
            provider = self.provider
            if provider is None:
                return fallback()

            raw = provider.generate(prompt, system_prompt=system, temperature=0.3)
            # Strip markdown fences if present
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned.strip())
            return json.loads(cleaned)
        except Exception:
            return fallback()

    # ------------------------------------------------------------------
    # Rule-based fallbacks (no API keys required)
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_based_resume_analyze(resume_text: str) -> dict[str, Any]:
        """Deterministic resume analysis using keyword matching."""
        text_lower = resume_text.lower()

        # Extract skills by keyword matching
        skill_keywords = [
            "python", "javascript", "typescript", "java", "go", "rust", "c++",
            "react", "vue", "angular", "node.js", "fastapi", "django", "flask",
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
            "sql", "postgresql", "mongodb", "redis", "elasticsearch",
            "machine learning", "deep learning", "nlp", "computer vision",
            "git", "ci/cd", "agile", "scrum", "leadership", "communication",
        ]
        skills = [s for s in skill_keywords if s in text_lower]

        # Estimate experience from year mentions
        year_matches = re.findall(r"(\d+)\+?\s*years?", text_lower)
        experience_years = max((int(y) for y in year_matches), default=0)

        # Detect education level
        education = "unknown"
        edu_patterns = [
            ("phd", "PhD"), ("ph.d", "PhD"), ("doctorate", "PhD"),
            ("master", "Master"), ("m.s.", "MS"), ("m.sc", "MS"),
            (" ms ", "MS"), ("ms in ", "MS"),
            ("bachelor", "Bachelor"), ("b.s.", "BS"), ("b.sc", "BS"),
        ]
        for pattern, label in edu_patterns:
            if pattern in text_lower:
                education = label
                break

        # Score based on skills and experience
        fit_score = min(1.0, (len(skills) * 0.08) + (experience_years * 0.05))

        return {
            "skills": skills,
            "experience_years": experience_years,
            "education": education,
            "fit_score": round(fit_score, 2),
            "summary": f"Candidate with {len(skills)} identified skills and {experience_years} years of experience.",
        }

    @staticmethod
    def _rule_based_job_match(
        candidate_profile: dict[str, Any],
        job_requirements: dict[str, Any],
    ) -> dict[str, Any]:
        """Deterministic job matching using skill overlap."""
        candidate_skills = {s.lower() for s in candidate_profile.get("skills", [])}
        required_skills = {s.lower() for s in job_requirements.get("required_skills", [])}

        if not required_skills:
            return {
                "match_score": 0.5,
                "matched_skills": list(candidate_skills),
                "missing_skills": [],
                "reasoning": "No specific skills required; candidate has general qualifications.",
            }

        matched = candidate_skills & required_skills
        missing = required_skills - candidate_skills
        score = len(matched) / len(required_skills) if required_skills else 0.0

        # Bonus for experience
        min_exp = job_requirements.get("min_experience", 0)
        cand_exp = candidate_profile.get("experience_years", 0)
        if min_exp > 0 and cand_exp >= min_exp:
            score = min(1.0, score + 0.1)

        return {
            "match_score": round(score, 2),
            "matched_skills": sorted(matched),
            "missing_skills": sorted(missing),
            "reasoning": (
                f"Candidate matches {len(matched)}/{len(required_skills)} required skills. "
                f"{'Meets' if cand_exp >= min_exp else 'Does not meet'} "
                f"experience requirement ({cand_exp} vs {min_exp} years)."
            ),
        }

    @staticmethod
    def _rule_based_interview_questions(job_posting: str, resume: str) -> list[str]:
        """Generate generic interview questions using keyword detection."""
        text = (job_posting + " " + resume).lower()

        questions = [
            "Tell me about your most challenging project and how you overcame obstacles.",
        ]

        if any(kw in text for kw in ["python", "javascript", "java", "code", "software"]):
            questions.append("Describe your approach to writing maintainable and testable code.")
        else:
            questions.append("What methodologies do you use to ensure quality in your work?")

        if any(kw in text for kw in ["lead", "manage", "team", "mentor"]):
            questions.append("How do you handle conflicting priorities when leading a team?")
        else:
            questions.append("Describe a situation where you collaborated effectively with a team.")

        if any(kw in text for kw in ["cloud", "aws", "azure", "gcp", "devops"]):
            questions.append("Walk me through your experience with cloud infrastructure and deployment.")
        else:
            questions.append("What tools and technologies are you most excited about learning?")

        if any(kw in text for kw in ["ml", "machine learning", "ai", "data"]):
            questions.append("How do you approach model evaluation and ensure fairness in AI systems?")
        else:
            questions.append("Where do you see your career heading in the next 3-5 years?")

        return questions[:5]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: AzureLLMClient | None = None


def get_llm_client() -> AzureLLMClient:
    """Get or create the global LLM client singleton."""
    global _client
    if _client is None:
        _client = AzureLLMClient()
    return _client


def reset_llm_client() -> AzureLLMClient:
    """Reset the global LLM client (for testing)."""
    global _client
    _client = AzureLLMClient()
    return _client


__all__ = [
    "AzureLLMClient",
    "get_llm_client",
    "reset_llm_client",
]
