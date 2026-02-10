"""REST API endpoints for the Agent Marketplace.

Endpoints:
- GET  /marketplace/agents          — list available agents
- GET  /marketplace/agents/{id}     — agent details + pricing
- POST /marketplace/hire            — hire an agent (triggers x402 payment)
- GET  /marketplace/jobs            — list jobs with payment status
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.marketplace import AgentListing, marketplace
from src.marketplace.x402 import (
    X402PaymentGate,
    PaymentConfig,
    PaymentProof,
    AgentEscrow,
)
from src.marketplace.hiring import HiringManager, HireRequest, BudgetTracker


# ── Router ──────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/marketplace", tags=["marketplace"])

# Module-level singletons (can be replaced in tests)
_escrow = AgentEscrow()
_budget = BudgetTracker(total_budget=100.0)
_hiring_manager = HiringManager(
    registry=marketplace,
    escrow=_escrow,
    budget_tracker=_budget,
)
_payment_gate = X402PaymentGate(
    PaymentConfig(
        pay_to="0x7483a9F237cf8043704D6b17DA31c12BfFF860DD",
        price=0.0,  # $0.00 for testing
    )
)


# ── Request / Response models ───────────────────────────────────────────────


class AgentListingResponse(BaseModel):
    agent_id: str
    name: str
    description: str
    skills: list[str]
    pricing_model: str
    price_per_unit: float
    price_display: str
    rating: float
    total_jobs: int
    endpoint: str
    protocol: str


class HireRequestBody(BaseModel):
    description: str = Field(..., min_length=1, max_length=2000)
    required_skills: list[str] = Field(default_factory=list)
    budget: float = Field(default=1.0, gt=0, le=1000)


class HireResponse(BaseModel):
    task_id: str
    status: str
    agent_id: str
    agent_name: str
    agreed_price: float
    escrow_id: str
    task_result: dict[str, Any] | None = None
    error: str = ""
    budget_remaining: float


class JobResponse(BaseModel):
    task_id: str
    status: str
    agent_name: str
    agreed_price: float
    escrow_id: str
    escrow_status: str
    error: str = ""


# ── Helpers ─────────────────────────────────────────────────────────────────


def _listing_to_response(listing: AgentListing) -> AgentListingResponse:
    return AgentListingResponse(
        agent_id=listing.agent_id,
        name=listing.name,
        description=listing.description,
        skills=listing.skills,
        pricing_model=listing.pricing_model,
        price_per_unit=listing.price_per_unit,
        price_display=listing.price_display,
        rating=listing.rating,
        total_jobs=listing.total_jobs,
        endpoint=listing.endpoint,
        protocol=listing.protocol,
    )


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.get("/agents", response_model=list[AgentListingResponse])
async def list_marketplace_agents(
    skill: str | None = None,
    max_price: float | None = None,
):
    """List available agents, optionally filtered by skill and max price."""
    if skill:
        agents = marketplace.discover_agents(skill, max_price)
    else:
        agents = marketplace.list_all()
        if max_price is not None:
            agents = [a for a in agents if a.price_per_unit <= max_price]
    return [_listing_to_response(a) for a in agents]


@router.get("/agents/{agent_id}", response_model=AgentListingResponse)
async def get_marketplace_agent(agent_id: str):
    """Get details for a specific agent."""
    listing = marketplace.get_agent(agent_id)
    if listing is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return _listing_to_response(listing)


@router.post("/hire", response_model=HireResponse, status_code=200)
async def hire_agent(body: HireRequestBody):
    """Hire an agent for a task. Triggers x402 escrow payment flow."""
    request = HireRequest(
        description=body.description,
        required_skills=body.required_skills,
        budget=body.budget,
    )

    result = _hiring_manager.hire(request)

    return HireResponse(
        task_id=result.task_id,
        status=result.status,
        agent_id=result.agent_id,
        agent_name=result.agent_name,
        agreed_price=result.agreed_price,
        escrow_id=result.escrow_id,
        task_result=result.task_result,
        error=result.error,
        budget_remaining=result.budget_remaining,
    )


@router.get("/jobs", response_model=list[JobResponse])
async def list_jobs():
    """List all hiring jobs with payment status."""
    jobs = []
    for hire in _hiring_manager.hire_history:
        escrow_status = "none"
        if hire.escrow_id:
            entry = _escrow.get_entry(hire.escrow_id)
            escrow_status = entry.status if entry else "unknown"
        jobs.append(JobResponse(
            task_id=hire.task_id,
            status=hire.status,
            agent_name=hire.agent_name,
            agreed_price=hire.agreed_price,
            escrow_id=hire.escrow_id,
            escrow_status=escrow_status,
            error=hire.error,
        ))
    return jobs


@router.get("/budget")
async def marketplace_budget():
    """Get current marketplace budget status."""
    return _budget.spending_report()


@router.get("/x402")
async def x402_info():
    """Get x402 payment gate information for this marketplace."""
    return _payment_gate.create_402_response(
        resource="/marketplace/hire",
        description="Payment required to hire agents from the marketplace",
    )
