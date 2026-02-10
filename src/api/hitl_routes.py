"""REST API endpoints for Human-in-the-Loop (HITL) approval workflow.

Endpoints:
- GET  /approvals/pending    — list pending approval requests
- GET  /approvals/stats      — approval statistics
- GET  /approvals/{id}/status — check approval status
- POST /approvals/{id}/approve — approve a request
- POST /approvals/{id}/reject  — reject a request
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.hitl import get_approval_gate, ApprovalStatus

router = APIRouter(tags=["hitl"])


# ── Request / Response models ───────────────────────────────────────────────


class ApprovalActionBody(BaseModel):
    reviewer: str = Field(default="human", max_length=200)
    reason: str = Field(default="", max_length=2000)


class ApprovalResponse(BaseModel):
    request_id: str
    action: str
    description: str
    cost_usdc: float
    status: str
    requester: str
    reviewer: str
    review_reason: str
    created_at: float
    reviewed_at: float
    expires_at: float
    details: dict[str, Any] = {}


class ApprovalStatsResponse(BaseModel):
    total_requests: int
    approved: int
    rejected: int
    auto_approved: int
    expired: int
    pending: int
    cost_threshold: float
    timeout_seconds: int


# ── Helpers ─────────────────────────────────────────────────────────────────


def _request_to_response(req) -> ApprovalResponse:
    return ApprovalResponse(
        request_id=req.request_id,
        action=req.action,
        description=req.description,
        cost_usdc=req.cost_usdc,
        status=req.status.value if hasattr(req.status, "value") else req.status,
        requester=req.requester,
        reviewer=req.reviewer,
        review_reason=req.review_reason,
        created_at=req.created_at,
        reviewed_at=req.reviewed_at,
        expires_at=req.expires_at,
        details=req.details,
    )


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.get("/approvals/pending", response_model=list[ApprovalResponse])
async def list_pending_approvals():
    """List all pending approval requests."""
    gate = get_approval_gate()
    pending = gate.list_pending()
    return [_request_to_response(r) for r in pending]


@router.get("/approvals/all", response_model=list[ApprovalResponse])
async def list_all_approvals():
    """List all approval requests (any status)."""
    gate = get_approval_gate()
    return [_request_to_response(r) for r in gate.list_all()]


@router.get("/approvals/stats", response_model=ApprovalStatsResponse)
async def approval_stats():
    """Get approval gate statistics."""
    gate = get_approval_gate()
    stats = gate.get_stats()
    return ApprovalStatsResponse(**stats)


@router.get("/approvals/{request_id}/status")
async def check_approval_status(request_id: str):
    """Check the status of a specific approval request."""
    gate = get_approval_gate()
    try:
        status = gate.check_approval(request_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Approval request '{request_id}' not found")

    req = gate.get_request(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail=f"Approval request '{request_id}' not found")

    return _request_to_response(req)


@router.post("/approvals/{request_id}/approve", response_model=ApprovalResponse)
async def approve_request(request_id: str, body: ApprovalActionBody | None = None):
    """Approve a pending approval request."""
    gate = get_approval_gate()
    reviewer = body.reviewer if body else "human"
    reason = body.reason if body else ""

    try:
        req = gate.approve(request_id, reviewer=reviewer, reason=reason)
    except ValueError as e:
        msg = str(e)
        if "Unknown" in msg:
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)

    return _request_to_response(req)


@router.post("/approvals/{request_id}/reject", response_model=ApprovalResponse)
async def reject_request(request_id: str, body: ApprovalActionBody | None = None):
    """Reject a pending approval request."""
    gate = get_approval_gate()
    reviewer = body.reviewer if body else "human"
    reason = body.reason if body else ""

    try:
        req = gate.reject(request_id, reviewer=reviewer, reason=reason)
    except ValueError as e:
        msg = str(e)
        if "Unknown" in msg:
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)

    return _request_to_response(req)
