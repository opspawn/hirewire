"""Tests for the Human-in-the-Loop (HITL) approval gate module.

Tests cover:
- ApprovalGate creation and configuration
- approval_required threshold logic
- request_approval lifecycle
- auto_approve for low-cost actions
- approve / reject flow
- expiration handling
- process_action convenience method
- statistics tracking
- list_pending / list_all
- edge cases and error handling
"""

import time

import pytest

from src.hitl import (
    ApprovalGate,
    ApprovalRequest,
    ApprovalStatus,
    get_approval_gate,
    reset_approval_gate,
)


@pytest.fixture
def gate():
    """Create a fresh ApprovalGate for each test."""
    return ApprovalGate(cost_threshold=1.0, timeout_seconds=3600)


# ── ApprovalGate creation ──────────────────────────────────────────────────


class TestApprovalGateCreation:
    def test_default_threshold(self, gate):
        assert gate.cost_threshold == 1.0

    def test_default_timeout(self, gate):
        assert gate.timeout_seconds == 3600

    def test_custom_threshold(self):
        g = ApprovalGate(cost_threshold=5.0)
        assert g.cost_threshold == 5.0

    def test_custom_timeout(self):
        g = ApprovalGate(timeout_seconds=60)
        assert g.timeout_seconds == 60

    def test_threshold_setter(self, gate):
        gate.cost_threshold = 10.0
        assert gate.cost_threshold == 10.0

    def test_timeout_setter(self, gate):
        gate.timeout_seconds = 120
        assert gate.timeout_seconds == 120


# ── approval_required ──────────────────────────────────────────────────────


class TestApprovalRequired:
    def test_below_threshold_not_required(self, gate):
        assert gate.approval_required("hire", 0.50) is False

    def test_at_threshold_not_required(self, gate):
        assert gate.approval_required("hire", 1.0) is False

    def test_above_threshold_required(self, gate):
        assert gate.approval_required("hire", 1.01) is True

    def test_zero_cost_not_required(self, gate):
        assert gate.approval_required("hire", 0.0) is False

    def test_high_cost_required(self, gate):
        assert gate.approval_required("expensive_op", 100.0) is True

    def test_custom_threshold(self):
        g = ApprovalGate(cost_threshold=0.5)
        assert g.approval_required("hire", 0.51) is True
        assert g.approval_required("hire", 0.50) is False


# ── request_approval ───────────────────────────────────────────────────────


class TestRequestApproval:
    def test_creates_request(self, gate):
        rid = gate.request_approval("hire_agent", cost_usdc=5.0)
        assert rid.startswith("apr_")

    def test_request_is_pending(self, gate):
        rid = gate.request_approval("hire_agent", cost_usdc=5.0)
        assert gate.check_approval(rid) == ApprovalStatus.PENDING

    def test_request_has_details(self, gate):
        rid = gate.request_approval(
            "hire_agent",
            details={"agent": "builder"},
            cost_usdc=5.0,
            description="Hire builder agent",
        )
        req = gate.get_request(rid)
        assert req is not None
        assert req.details == {"agent": "builder"}
        assert req.cost_usdc == 5.0
        assert req.description == "Hire builder agent"

    def test_request_has_expiry(self, gate):
        rid = gate.request_approval("hire_agent", cost_usdc=5.0)
        req = gate.get_request(rid)
        assert req.expires_at > req.created_at

    def test_multiple_requests(self, gate):
        r1 = gate.request_approval("hire_1", cost_usdc=2.0)
        r2 = gate.request_approval("hire_2", cost_usdc=3.0)
        assert r1 != r2
        assert gate.check_approval(r1) == ApprovalStatus.PENDING
        assert gate.check_approval(r2) == ApprovalStatus.PENDING


# ── auto_approve ───────────────────────────────────────────────────────────


class TestAutoApprove:
    def test_auto_approve_returns_id(self, gate):
        rid = gate.auto_approve("low_cost_action", cost_usdc=0.10)
        assert rid.startswith("apr_")

    def test_auto_approve_status(self, gate):
        rid = gate.auto_approve("low_cost_action", cost_usdc=0.10)
        assert gate.check_approval(rid) == ApprovalStatus.AUTO_APPROVED

    def test_auto_approve_has_reviewer(self, gate):
        rid = gate.auto_approve("low_cost_action")
        req = gate.get_request(rid)
        assert req.reviewer == "auto"

    def test_auto_approve_tracked_in_stats(self, gate):
        gate.auto_approve("action_1")
        gate.auto_approve("action_2")
        stats = gate.get_stats()
        assert stats["auto_approved"] == 2


# ── approve / reject ──────────────────────────────────────────────────────


class TestApproveReject:
    def test_approve_pending(self, gate):
        rid = gate.request_approval("hire", cost_usdc=5.0)
        req = gate.approve(rid, reviewer="admin", reason="Looks good")
        assert req.status == ApprovalStatus.APPROVED
        assert req.reviewer == "admin"
        assert req.review_reason == "Looks good"
        assert req.reviewed_at > 0

    def test_reject_pending(self, gate):
        rid = gate.request_approval("hire", cost_usdc=5.0)
        req = gate.reject(rid, reviewer="admin", reason="Too expensive")
        assert req.status == ApprovalStatus.REJECTED
        assert req.reviewer == "admin"

    def test_cannot_approve_already_approved(self, gate):
        rid = gate.request_approval("hire", cost_usdc=5.0)
        gate.approve(rid)
        with pytest.raises(ValueError, match="Cannot approve"):
            gate.approve(rid)

    def test_cannot_reject_already_rejected(self, gate):
        rid = gate.request_approval("hire", cost_usdc=5.0)
        gate.reject(rid)
        with pytest.raises(ValueError, match="Cannot reject"):
            gate.reject(rid)

    def test_cannot_approve_auto_approved(self, gate):
        rid = gate.auto_approve("action")
        with pytest.raises(ValueError, match="Cannot approve"):
            gate.approve(rid)

    def test_approve_unknown_request(self, gate):
        with pytest.raises(ValueError, match="Unknown"):
            gate.approve("nonexistent_id")

    def test_reject_unknown_request(self, gate):
        with pytest.raises(ValueError, match="Unknown"):
            gate.reject("nonexistent_id")


# ── Expiration ─────────────────────────────────────────────────────────────


class TestExpiration:
    def test_expired_request(self):
        gate = ApprovalGate(cost_threshold=0.0, timeout_seconds=0)
        rid = gate.request_approval("hire", cost_usdc=5.0)
        # Force expiry by setting expires_at to past
        req = gate.get_request(rid)
        req.expires_at = time.time() - 1
        status = gate.check_approval(rid)
        assert status == ApprovalStatus.EXPIRED

    def test_cannot_approve_expired(self):
        gate = ApprovalGate(cost_threshold=0.0, timeout_seconds=0)
        rid = gate.request_approval("hire", cost_usdc=5.0)
        req = gate.get_request(rid)
        req.expires_at = time.time() - 1
        with pytest.raises(ValueError, match="expired"):
            gate.approve(rid)

    def test_expired_not_in_pending_list(self):
        gate = ApprovalGate(cost_threshold=0.0, timeout_seconds=0)
        rid = gate.request_approval("hire", cost_usdc=5.0)
        req = gate.get_request(rid)
        req.expires_at = time.time() - 1
        pending = gate.list_pending()
        assert len(pending) == 0


# ── process_action ─────────────────────────────────────────────────────────


class TestProcessAction:
    def test_below_threshold_auto_approves(self, gate):
        rid, requires_wait = gate.process_action("hire", cost_usdc=0.50)
        assert requires_wait is False
        assert gate.check_approval(rid) == ApprovalStatus.AUTO_APPROVED

    def test_above_threshold_requires_wait(self, gate):
        rid, requires_wait = gate.process_action("hire", cost_usdc=5.0)
        assert requires_wait is True
        assert gate.check_approval(rid) == ApprovalStatus.PENDING

    def test_at_threshold_auto_approves(self, gate):
        rid, requires_wait = gate.process_action("hire", cost_usdc=1.0)
        assert requires_wait is False


# ── Statistics ─────────────────────────────────────────────────────────────


class TestStatistics:
    def test_initial_stats(self, gate):
        stats = gate.get_stats()
        assert stats["total_requests"] == 0
        assert stats["approved"] == 0
        assert stats["rejected"] == 0

    def test_stats_after_operations(self, gate):
        r1 = gate.request_approval("a", cost_usdc=5.0)
        gate.approve(r1)
        r2 = gate.request_approval("b", cost_usdc=5.0)
        gate.reject(r2)
        gate.auto_approve("c", cost_usdc=0.10)

        stats = gate.get_stats()
        assert stats["total_requests"] == 3
        assert stats["approved"] == 1
        assert stats["rejected"] == 1
        assert stats["auto_approved"] == 1
        assert stats["pending"] == 0

    def test_stats_include_config(self, gate):
        stats = gate.get_stats()
        assert stats["cost_threshold"] == 1.0
        assert stats["timeout_seconds"] == 3600


# ── Listing ────────────────────────────────────────────────────────────────


class TestListing:
    def test_list_pending(self, gate):
        r1 = gate.request_approval("a", cost_usdc=5.0)
        r2 = gate.request_approval("b", cost_usdc=3.0)
        gate.approve(r1)
        pending = gate.list_pending()
        assert len(pending) == 1
        assert pending[0].request_id == r2

    def test_list_all(self, gate):
        gate.request_approval("a", cost_usdc=5.0)
        gate.auto_approve("b", cost_usdc=0.10)
        all_reqs = gate.list_all()
        assert len(all_reqs) == 2

    def test_list_pending_sorted_by_time(self, gate):
        r1 = gate.request_approval("first", cost_usdc=2.0)
        r2 = gate.request_approval("second", cost_usdc=3.0)
        pending = gate.list_pending()
        assert pending[0].request_id == r2  # newest first


# ── get_request / clear ────────────────────────────────────────────────────


class TestGetRequestAndClear:
    def test_get_existing(self, gate):
        rid = gate.request_approval("action", cost_usdc=5.0)
        req = gate.get_request(rid)
        assert req is not None
        assert req.action == "action"

    def test_get_nonexistent(self, gate):
        assert gate.get_request("nonexistent") is None

    def test_clear(self, gate):
        gate.request_approval("a", cost_usdc=5.0)
        gate.clear()
        assert len(gate.list_all()) == 0
        assert gate.get_stats()["total_requests"] == 0


# ── ApprovalRequest dataclass ──────────────────────────────────────────────


class TestApprovalRequest:
    def test_to_dict(self):
        req = ApprovalRequest(action="test", cost_usdc=1.0)
        d = req.to_dict()
        assert d["action"] == "test"
        assert d["status"] == "pending"

    def test_is_expired_false(self):
        req = ApprovalRequest(expires_at=time.time() + 3600)
        assert req.is_expired is False

    def test_is_expired_true(self):
        req = ApprovalRequest(expires_at=time.time() - 1)
        assert req.is_expired is True

    def test_is_actionable(self):
        req = ApprovalRequest(
            status=ApprovalStatus.PENDING,
            expires_at=time.time() + 3600,
        )
        assert req.is_actionable is True

    def test_not_actionable_when_approved(self):
        req = ApprovalRequest(
            status=ApprovalStatus.APPROVED,
            expires_at=time.time() + 3600,
        )
        assert req.is_actionable is False


# ── Singleton ──────────────────────────────────────────────────────────────


class TestSingleton:
    def test_get_approval_gate(self):
        gate = get_approval_gate()
        assert isinstance(gate, ApprovalGate)

    def test_reset_approval_gate(self):
        old = get_approval_gate()
        new = reset_approval_gate()
        assert old is not new
