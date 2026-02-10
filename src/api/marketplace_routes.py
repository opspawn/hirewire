"""REST API endpoints for the Agent Marketplace and x402 Payments.

Marketplace endpoints:
- GET  /marketplace/agents          — list available agents with filters
- POST /marketplace/agents          — register an agent listing
- GET  /marketplace/agents/{id}     — agent details + reputation
- POST /marketplace/hire            — hire an agent (triggers x402 payment)
- GET  /marketplace/hire/{id}/status — hiring status
- GET  /marketplace/jobs            — list jobs with payment status
- GET  /marketplace/budget          — budget status

Payment endpoints:
- POST /payments/request            — create payment request (returns 402)
- POST /payments/verify             — verify payment receipt
- GET  /payments/balance/{agent_id} — agent balance
- GET  /payments/ledger             — transaction history
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.marketplace import AgentListing, marketplace
from src.marketplace.x402 import (
    X402PaymentGate,
    PaymentConfig,
    PaymentProof,
    AgentEscrow,
    PaymentManager,
    PaymentLedger,
    payment_manager,
)
from src.marketplace.hiring import HiringManager, HireRequest, BudgetTracker


# ── Router ──────────────────────────────────────────────────────────────────

router = APIRouter(tags=["marketplace"])

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
_payment_manager = PaymentManager(
    gate=_payment_gate,
    escrow=_escrow,
    ledger=PaymentLedger(),
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
    completed_jobs: int = 0
    failed_jobs: int = 0
    completion_rate: float = 0.0
    total_earnings: float = 0.0
    availability: str = "available"
    endpoint: str
    protocol: str


class AgentRegistrationBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    skills: list[str] = Field(default_factory=list)
    pricing_model: str = Field(default="per-task")
    price_per_unit: float = Field(default=0.0, ge=0)
    endpoint: str = Field(default="")
    protocol: str = Field(default="a2a")


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


class HireStatusResponse(BaseModel):
    task_id: str
    status: str
    agent_id: str
    agent_name: str
    agreed_price: float
    escrow_id: str
    escrow_status: str = ""
    error: str = ""


class JobResponse(BaseModel):
    task_id: str
    status: str
    agent_name: str
    agreed_price: float
    escrow_id: str
    escrow_status: str
    error: str = ""


class PaymentRequestBody(BaseModel):
    resource: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)
    payee: str = Field(..., min_length=1)
    description: str = Field(default="Payment required")


class PaymentVerifyBody(BaseModel):
    payer: str = Field(..., min_length=1)
    payee: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)
    tx_hash: str = Field(default="")
    network: str = Field(default="eip155:8453")


class BalanceResponse(BaseModel):
    agent_id: str
    balance: float


class LedgerEntryResponse(BaseModel):
    entry_id: str
    event_type: str
    payer: str
    payee: str
    amount: float
    task_id: str
    escrow_id: str
    payment_id: str
    network: str
    timestamp: float


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
        completed_jobs=listing.completed_jobs,
        failed_jobs=listing.failed_jobs,
        completion_rate=listing.completion_rate,
        total_earnings=listing.total_earnings,
        availability=listing.availability,
        endpoint=listing.endpoint,
        protocol=listing.protocol,
    )


# ── Marketplace Endpoints ──────────────────────────────────────────────────


@router.get("/marketplace/agents", response_model=list[AgentListingResponse])
async def list_marketplace_agents(
    skill: str | None = None,
    max_price: float | None = None,
    sort_by: str | None = None,
    available_only: bool = False,
):
    """List available agents, optionally filtered by skill, max price, and sorted."""
    if skill:
        agents = marketplace.discover_agents(skill, max_price)
    else:
        agents = marketplace.list_all()
        if max_price is not None:
            agents = [a for a in agents if a.price_per_unit <= max_price]

    if available_only:
        agents = [a for a in agents if a.availability == "available"]

    if sort_by == "price":
        agents.sort(key=lambda a: a.price_per_unit)
    elif sort_by == "rating":
        agents.sort(key=lambda a: a.rating, reverse=True)
    elif sort_by == "jobs":
        agents.sort(key=lambda a: a.total_jobs, reverse=True)

    return [_listing_to_response(a) for a in agents]


@router.post("/marketplace/agents", response_model=AgentListingResponse, status_code=201)
async def register_marketplace_agent(body: AgentRegistrationBody):
    """Register a new agent in the marketplace."""
    listing = AgentListing(
        name=body.name,
        description=body.description,
        skills=body.skills,
        pricing_model=body.pricing_model,
        price_per_unit=body.price_per_unit,
        endpoint=body.endpoint,
        protocol=body.protocol,
    )
    marketplace.register_agent(listing)
    return _listing_to_response(listing)


@router.get("/marketplace/agents/{agent_id}", response_model=AgentListingResponse)
async def get_marketplace_agent(agent_id: str):
    """Get details for a specific agent including reputation."""
    listing = marketplace.get_agent(agent_id)
    if listing is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return _listing_to_response(listing)


@router.post("/marketplace/hire", response_model=HireResponse, status_code=200)
async def hire_agent(body: HireRequestBody):
    """Hire an agent for a task. Triggers x402 escrow payment flow.

    If the budget exceeds the HITL cost threshold, an approval request
    is created and must be approved before the hire proceeds.
    """
    from src.hitl import get_approval_gate

    gate = get_approval_gate()
    request = HireRequest(
        description=body.description,
        required_skills=body.required_skills,
        budget=body.budget,
    )

    # HITL gate: check if approval is required
    approval_id, requires_wait = gate.process_action(
        action="marketplace_hire",
        cost_usdc=body.budget,
        details={
            "description": body.description,
            "required_skills": body.required_skills,
            "budget": body.budget,
        },
        description=f"Hire agent for: {body.description[:100]}",
        requester="marketplace",
    )

    result = _hiring_manager.hire(request)

    response = HireResponse(
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

    return response


@router.get("/marketplace/hire/{task_id}/status", response_model=HireStatusResponse)
async def get_hire_status(task_id: str):
    """Get the status of a hiring request."""
    for hire in _hiring_manager.hire_history:
        if hire.task_id == task_id:
            escrow_status = "none"
            if hire.escrow_id:
                entry = _escrow.get_entry(hire.escrow_id)
                escrow_status = entry.status if entry else "unknown"
            return HireStatusResponse(
                task_id=hire.task_id,
                status=hire.status,
                agent_id=hire.agent_id,
                agent_name=hire.agent_name,
                agreed_price=hire.agreed_price,
                escrow_id=hire.escrow_id,
                escrow_status=escrow_status,
                error=hire.error,
            )
    raise HTTPException(status_code=404, detail=f"Hiring request '{task_id}' not found")


@router.get("/marketplace/jobs", response_model=list[JobResponse])
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


@router.get("/marketplace/budget")
async def marketplace_budget():
    """Get current marketplace budget status."""
    return _budget.spending_report()


@router.get("/marketplace/x402")
async def x402_info():
    """Get x402 payment gate information for this marketplace."""
    return _payment_gate.create_402_response(
        resource="/marketplace/hire",
        description="Payment required to hire agents from the marketplace",
    )


# ── Payment Endpoints ─────────────────────────────────────────────────────


@router.post("/payments/request")
async def create_payment_request(body: PaymentRequestBody):
    """Create an x402 payment request (returns 402 response body)."""
    resp = _payment_manager.create_payment_request(
        resource=body.resource,
        amount=body.amount,
        payee=body.payee,
        description=body.description,
    )
    return JSONResponse(
        content=resp,
        status_code=402,
        headers={"X-Payment": "required"},
    )


@router.post("/payments/verify")
async def verify_payment(body: PaymentVerifyBody):
    """Verify a payment receipt."""
    proof = PaymentProof(
        payer=body.payer,
        payee=body.payee,
        amount=body.amount,
        tx_hash=body.tx_hash,
        network=body.network,
    )
    ok = _payment_manager.verify_payment(proof)
    return {
        "verified": ok,
        "payment_id": proof.payment_id,
        "amount": proof.amount,
        "network": proof.network,
    }


@router.get("/payments/balance/{agent_id}", response_model=BalanceResponse)
async def get_agent_balance(agent_id: str):
    """Get an agent's current balance."""
    balance = _payment_manager.get_balance(agent_id)
    return BalanceResponse(agent_id=agent_id, balance=balance)


@router.get("/payments/ledger", response_model=list[LedgerEntryResponse])
async def get_payment_ledger(
    event_type: str | None = None,
    agent_id: str | None = None,
    task_id: str | None = None,
):
    """Get the full payment ledger with optional filters."""
    entries = _payment_manager.ledger.get_entries(
        event_type=event_type,
        agent_id=agent_id,
        task_id=task_id,
    )
    return [
        LedgerEntryResponse(
            entry_id=e.entry_id,
            event_type=e.event_type,
            payer=e.payer,
            payee=e.payee,
            amount=e.amount,
            task_id=e.task_id,
            escrow_id=e.escrow_id,
            payment_id=e.payment_id,
            network=e.network,
            timestamp=e.timestamp,
        )
        for e in entries
    ]
