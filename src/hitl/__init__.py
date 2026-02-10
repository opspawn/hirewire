"""Human-in-the-Loop (HITL) Approval Gate for HireWire.

Provides an approval workflow for expensive operations in the hiring pipeline.
When a task or hiring action exceeds a configurable cost threshold, it requires
explicit human approval before proceeding.

Configuration:
    HITL_COST_THRESHOLD: Minimum cost (USDC) requiring approval (default: $1.00)
    HITL_TIMEOUT_SECONDS: How long an approval request stays valid (default: 3600)
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    AUTO_APPROVED = "auto_approved"


@dataclass
class ApprovalRequest:
    """An approval request for a human-in-the-loop gate."""

    request_id: str = field(default_factory=lambda: f"apr_{uuid.uuid4().hex[:12]}")
    action: str = ""
    description: str = ""
    cost_usdc: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    requester: str = "system"
    reviewer: str = ""
    review_reason: str = ""
    created_at: float = field(default_factory=time.time)
    reviewed_at: float = 0.0
    expires_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @property
    def is_expired(self) -> bool:
        return self.expires_at > 0 and time.time() > self.expires_at

    @property
    def is_actionable(self) -> bool:
        return self.status == ApprovalStatus.PENDING and not self.is_expired


class ApprovalGate:
    """Human-in-the-Loop approval gate.

    Checks whether an action requires approval based on cost threshold
    and manages the approval lifecycle.
    """

    def __init__(
        self,
        cost_threshold: float | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self._cost_threshold = cost_threshold if cost_threshold is not None else float(
            os.environ.get("HITL_COST_THRESHOLD", "1.0")
        )
        self._timeout_seconds = timeout_seconds if timeout_seconds is not None else int(
            os.environ.get("HITL_TIMEOUT_SECONDS", "3600")
        )
        self._requests: dict[str, ApprovalRequest] = {}
        self._stats = {
            "total_requests": 0,
            "approved": 0,
            "rejected": 0,
            "auto_approved": 0,
            "expired": 0,
            "pending": 0,
        }

    @property
    def cost_threshold(self) -> float:
        return self._cost_threshold

    @cost_threshold.setter
    def cost_threshold(self, value: float) -> None:
        self._cost_threshold = value

    @property
    def timeout_seconds(self) -> int:
        return self._timeout_seconds

    @timeout_seconds.setter
    def timeout_seconds(self, value: int) -> None:
        self._timeout_seconds = value

    def approval_required(self, action: str, cost: float) -> bool:
        """Check if an action requires human approval based on cost threshold.

        Returns True if cost exceeds the threshold and approval is needed.
        """
        return cost > self._cost_threshold

    def request_approval(
        self,
        action: str,
        details: dict[str, Any] | None = None,
        cost_usdc: float = 0.0,
        description: str = "",
        requester: str = "system",
    ) -> str:
        """Create an approval request. Returns the request_id."""
        now = time.time()
        req = ApprovalRequest(
            action=action,
            description=description or f"Approval required for: {action}",
            cost_usdc=cost_usdc,
            details=details or {},
            status=ApprovalStatus.PENDING,
            requester=requester,
            created_at=now,
            expires_at=now + self._timeout_seconds,
        )
        self._requests[req.request_id] = req
        self._stats["total_requests"] += 1
        self._stats["pending"] += 1
        return req.request_id

    def check_approval(self, request_id: str) -> ApprovalStatus:
        """Check the status of an approval request.

        Returns the current status. If the request has expired,
        automatically transitions to EXPIRED.
        """
        req = self._requests.get(request_id)
        if req is None:
            raise ValueError(f"Unknown approval request: {request_id}")

        # Auto-expire if past deadline
        if req.status == ApprovalStatus.PENDING and req.is_expired:
            req.status = ApprovalStatus.EXPIRED
            self._stats["pending"] = max(0, self._stats["pending"] - 1)
            self._stats["expired"] += 1

        return req.status

    def auto_approve(
        self,
        action: str,
        details: dict[str, Any] | None = None,
        cost_usdc: float = 0.0,
        description: str = "",
    ) -> str:
        """Auto-approve a low-cost action. Returns request_id for audit trail."""
        now = time.time()
        req = ApprovalRequest(
            action=action,
            description=description or f"Auto-approved: {action}",
            cost_usdc=cost_usdc,
            details=details or {},
            status=ApprovalStatus.AUTO_APPROVED,
            requester="system",
            reviewer="auto",
            review_reason="Below cost threshold",
            created_at=now,
            reviewed_at=now,
            expires_at=0.0,
        )
        self._requests[req.request_id] = req
        self._stats["total_requests"] += 1
        self._stats["auto_approved"] += 1
        return req.request_id

    def approve(
        self,
        request_id: str,
        reviewer: str = "human",
        reason: str = "",
    ) -> ApprovalRequest:
        """Approve a pending request."""
        req = self._requests.get(request_id)
        if req is None:
            raise ValueError(f"Unknown approval request: {request_id}")
        if req.status != ApprovalStatus.PENDING:
            raise ValueError(
                f"Cannot approve request in status '{req.status.value}'"
            )
        if req.is_expired:
            req.status = ApprovalStatus.EXPIRED
            self._stats["pending"] = max(0, self._stats["pending"] - 1)
            self._stats["expired"] += 1
            raise ValueError("Approval request has expired")

        req.status = ApprovalStatus.APPROVED
        req.reviewer = reviewer
        req.review_reason = reason
        req.reviewed_at = time.time()
        self._stats["pending"] = max(0, self._stats["pending"] - 1)
        self._stats["approved"] += 1
        return req

    def reject(
        self,
        request_id: str,
        reviewer: str = "human",
        reason: str = "",
    ) -> ApprovalRequest:
        """Reject a pending request."""
        req = self._requests.get(request_id)
        if req is None:
            raise ValueError(f"Unknown approval request: {request_id}")
        if req.status != ApprovalStatus.PENDING:
            raise ValueError(
                f"Cannot reject request in status '{req.status.value}'"
            )

        req.status = ApprovalStatus.REJECTED
        req.reviewer = reviewer
        req.review_reason = reason
        req.reviewed_at = time.time()
        self._stats["pending"] = max(0, self._stats["pending"] - 1)
        self._stats["rejected"] += 1
        return req

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Get an approval request by ID."""
        return self._requests.get(request_id)

    def list_pending(self) -> list[ApprovalRequest]:
        """List all pending (non-expired) approval requests."""
        pending = []
        for req in self._requests.values():
            if req.status == ApprovalStatus.PENDING:
                if req.is_expired:
                    req.status = ApprovalStatus.EXPIRED
                    self._stats["pending"] = max(0, self._stats["pending"] - 1)
                    self._stats["expired"] += 1
                else:
                    pending.append(req)
        return sorted(pending, key=lambda r: r.created_at, reverse=True)

    def list_all(self) -> list[ApprovalRequest]:
        """List all approval requests."""
        return sorted(
            self._requests.values(),
            key=lambda r: r.created_at,
            reverse=True,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get approval statistics."""
        return {
            **self._stats,
            "cost_threshold": self._cost_threshold,
            "timeout_seconds": self._timeout_seconds,
        }

    def process_action(
        self,
        action: str,
        cost_usdc: float,
        details: dict[str, Any] | None = None,
        description: str = "",
        requester: str = "system",
    ) -> tuple[str, bool]:
        """Process an action through the approval gate.

        Returns (request_id, requires_wait):
        - If below threshold: auto-approves and returns (id, False)
        - If above threshold: creates pending request and returns (id, True)
        """
        if self.approval_required(action, cost_usdc):
            request_id = self.request_approval(
                action=action,
                details=details,
                cost_usdc=cost_usdc,
                description=description,
                requester=requester,
            )
            return request_id, True
        else:
            request_id = self.auto_approve(
                action=action,
                details=details,
                cost_usdc=cost_usdc,
                description=description,
            )
            return request_id, False

    def clear(self) -> None:
        """Clear all requests and reset stats. For testing."""
        self._requests.clear()
        self._stats = {
            "total_requests": 0,
            "approved": 0,
            "rejected": 0,
            "auto_approved": 0,
            "expired": 0,
            "pending": 0,
        }


# Module-level singleton
_approval_gate: ApprovalGate | None = None


def get_approval_gate() -> ApprovalGate:
    """Get or create the global approval gate instance."""
    global _approval_gate
    if _approval_gate is None:
        _approval_gate = ApprovalGate()
    return _approval_gate


def reset_approval_gate() -> ApprovalGate:
    """Reset the global approval gate (for testing)."""
    global _approval_gate
    _approval_gate = ApprovalGate()
    return _approval_gate


__all__ = [
    "ApprovalGate",
    "ApprovalRequest",
    "ApprovalStatus",
    "get_approval_gate",
    "reset_approval_gate",
]
