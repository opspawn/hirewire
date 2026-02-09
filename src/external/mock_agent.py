"""Mock External Agent — simulates a 'designer' agent available for hire.

Runs as a standalone FastAPI app that implements A2A-style endpoints:
- POST /a2a/tasks  — accept a task, return a design deliverable
- GET  /agent-card — return the agent card with capabilities and pricing

Used for demos and tests; can be started in-process via ``create_mock_agent_app()``.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class TaskRequest(BaseModel):
    """Incoming task from a hiring agent."""
    task_id: str = ""
    description: str
    from_agent: str = "unknown"
    budget: float = 0.0


class TaskResult(BaseModel):
    """Result returned after completing a task."""
    task_id: str
    status: str  # "completed" | "failed"
    agent: str
    deliverable: str
    price_usdc: float
    completed_at: str


AGENT_CARD: dict[str, Any] = {
    "name": "designer-ext-001",
    "description": "Creates professional UI/UX designs, mockups, and design specifications",
    "version": "1.0.0",
    "skills": ["design", "ui", "ux", "mockup", "landing-page", "branding", "prototyping"],
    "price_per_call": "$0.05",
    "pricing": {
        "amount_usdc": 0.05,
        "model": "per-task",
        "currency": "USDC",
    },
    "endpoint": "",  # filled at startup
    "protocol": "a2a",
    "payment": "x402",
    "metadata": {
        "provider": "DesignStudio AI",
        "rating": 4.8,
        "tasks_completed": 142,
    },
}

# Keep track of tasks processed (for testing)
_processed_tasks: list[TaskResult] = []


def create_mock_agent_app(port: int = 9100) -> FastAPI:
    """Create the FastAPI application for the mock external designer agent."""

    app = FastAPI(title="Designer Agent (External)", version="1.0.0")

    # Update the endpoint in the agent card so callers know where to reach us
    AGENT_CARD["endpoint"] = f"http://127.0.0.1:{port}"

    @app.get("/agent-card")
    async def get_agent_card() -> dict[str, Any]:
        """Return this agent's capability card (A2A discovery)."""
        return AGENT_CARD

    @app.post("/a2a/tasks", response_model=TaskResult)
    async def handle_task(req: TaskRequest) -> TaskResult:
        """Accept and process a task, returning a design deliverable."""
        task_id = req.task_id or f"ext-{uuid.uuid4().hex[:8]}"

        # Simulate doing design work — produce a mock deliverable
        deliverable = _generate_design_deliverable(req.description)

        result = TaskResult(
            task_id=task_id,
            status="completed",
            agent="designer-ext-001",
            deliverable=deliverable,
            price_usdc=AGENT_CARD["pricing"]["amount_usdc"],
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        _processed_tasks.append(result)
        return result

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "agent": "designer-ext-001"}

    return app


def _generate_design_deliverable(description: str) -> str:
    """Generate a mock design deliverable based on the task description."""
    desc_lower = description.lower()

    if "landing" in desc_lower or "page" in desc_lower:
        return (
            "## Design Deliverable: Landing Page\n\n"
            "### Layout\n"
            "- Hero section with bold headline, sub-headline, and CTA button\n"
            "- Feature grid (3 columns): Agent Hiring, x402 Payments, Multi-Agent Orchestration\n"
            "- Social proof section with agent statistics\n"
            "- Footer with links and newsletter signup\n\n"
            "### Color Palette\n"
            "- Primary: #6C5CE7 (Electric Purple)\n"
            "- Secondary: #00D2D3 (Cyan)\n"
            "- Background: #0F0F1A (Dark Navy)\n"
            "- Text: #F5F6FA (Off-White)\n\n"
            "### Typography\n"
            "- Headlines: Inter Bold, 48px\n"
            "- Body: Inter Regular, 16px\n"
            "- Code: JetBrains Mono, 14px\n\n"
            "### Responsive Breakpoints\n"
            "- Desktop: 1200px+\n"
            "- Tablet: 768px-1199px\n"
            "- Mobile: <768px\n"
        )

    if "logo" in desc_lower or "brand" in desc_lower:
        return (
            "## Design Deliverable: Brand Identity\n\n"
            "### Logo\n"
            "- Geometric hexagon mark with embedded circuit motif\n"
            "- Wordmark: 'AgentOS' in Inter ExtraBold\n\n"
            "### Brand Colors\n"
            "- Primary: #6C5CE7, Secondary: #00D2D3, Accent: #FF6B6B\n"
        )

    # Generic design deliverable
    return (
        f"## Design Deliverable\n\n"
        f"### Task: {description}\n\n"
        "### Specifications\n"
        "- Clean, modern design following Material Design 3 principles\n"
        "- Responsive layout with mobile-first approach\n"
        "- Accessible (WCAG 2.1 AA compliant)\n"
        "- Dark mode support included\n\n"
        "### Assets Provided\n"
        "- Figma file with component library\n"
        "- CSS variables for theming\n"
        "- SVG icons and illustrations\n"
    )


def get_processed_tasks() -> list[TaskResult]:
    """Return all tasks this agent has processed (for testing)."""
    return list(_processed_tasks)


def clear_processed_tasks() -> None:
    """Clear processed task history (for testing)."""
    _processed_tasks.clear()
