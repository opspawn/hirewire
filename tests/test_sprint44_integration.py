"""Sprint 44 — Quality & Integration Polish tests.

Comprehensive tests for:
1. Full hiring pipeline end-to-end with storage persistence
2. Error handling edge cases (double release, double refund, timeout, etc.)
3. Concurrent hiring pipeline scenarios
4. HITL (Human-in-the-Loop) integration with hiring flow
5. API error code coverage (405 Method Not Allowed, content-type, etc.)
6. Pipeline data integrity and audit trail consistency
7. Marketplace registry edge cases
8. Payment Manager edge cases
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.marketplace import AgentListing, MarketplaceRegistry, SkillMatcher, marketplace
from src.marketplace.x402 import (
    PaymentConfig,
    PaymentProof,
    X402PaymentGate,
    EscrowEntry,
    AgentEscrow,
    LedgerEntry,
    PaymentLedger,
    PaymentManager,
)
from src.marketplace.hiring import HireRequest, HireResult, BudgetTracker, HiringManager
from src.hitl import ApprovalGate, ApprovalStatus, get_approval_gate, reset_approval_gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_listing(
    name: str = "TestAgent",
    skills: list[str] | None = None,
    price: float = 0.01,
    rating: float = 4.0,
    agent_id: str | None = None,
    availability: str = "available",
) -> AgentListing:
    listing = AgentListing(
        name=name,
        description=f"Test agent: {name}",
        skills=skills or ["code", "testing"],
        pricing_model="per-task",
        price_per_unit=price,
        rating=rating,
        endpoint=f"http://localhost:9000/{name.lower()}",
        availability=availability,
    )
    if agent_id:
        listing.agent_id = agent_id
    return listing


def _seeded_registry() -> MarketplaceRegistry:
    reg = MarketplaceRegistry()
    reg.register_agent(_make_listing("Coder", ["code", "python", "testing"], 0.02, 4.5, "coder-001"))
    reg.register_agent(_make_listing("Designer", ["design", "ui", "ux"], 0.05, 4.8, "designer-001"))
    reg.register_agent(_make_listing("Researcher", ["research", "analysis", "web-search"], 0.01, 4.2, "researcher-001"))
    reg.register_agent(_make_listing("DevOps", ["deployment", "docker", "ci-cd"], 0.03, 3.9, "devops-001"))
    reg.register_agent(_make_listing("Writer", ["writing", "docs", "content"], 0.00, 4.0, "writer-001"))
    return reg


def _create_pipeline(budget: float = 100.0, registry: MarketplaceRegistry | None = None):
    reg = registry or _seeded_registry()
    escrow = AgentEscrow()
    bt = BudgetTracker(total_budget=budget)
    manager = HiringManager(registry=reg, escrow=escrow, budget_tracker=bt)
    return manager, reg, escrow, bt


# ===================================================================
# 1. Full Pipeline with Storage Persistence
# ===================================================================


class TestPipelineWithStorage:
    """Verify the pipeline writes to SQLite storage correctly."""

    def test_task_persists_through_storage(self):
        """Task saved via storage can be retrieved."""
        from src.storage import get_storage

        storage = get_storage()
        task_id = "test-persist-001"
        storage.save_task(task_id, "Build REST API", workflow="hiring", budget_usd=5.0)

        retrieved = storage.get_task(task_id)
        assert retrieved is not None
        assert retrieved["description"] == "Build REST API"
        assert retrieved["budget_usd"] == 5.0
        assert retrieved["status"] == "pending"

    def test_task_status_updates_through_storage(self):
        """Task status transitions are persisted."""
        from src.storage import get_storage

        storage = get_storage()
        task_id = "test-status-001"
        storage.save_task(task_id, "Code review", workflow="hiring", budget_usd=2.0)

        storage.update_task_status(task_id, "running")
        assert storage.get_task(task_id)["status"] == "running"

        storage.update_task_status(task_id, "completed", result={"output": "done"})
        task = storage.get_task(task_id)
        assert task["status"] == "completed"

    def test_payment_persists_through_storage(self):
        """Payment records are persisted in storage."""
        from src.storage import get_storage

        storage = get_storage()
        storage.save_payment(
            tx_id="tx-001",
            from_agent="ceo",
            to_agent="coder",
            amount_usdc=0.05,
            task_id="task-001",
        )

        payments = storage.get_payments(task_id="task-001")
        assert len(payments) >= 1
        assert payments[0]["amount_usdc"] == 0.05

    def test_multiple_tasks_listed_by_status(self):
        """Storage lists tasks filtered by status."""
        from src.storage import get_storage

        storage = get_storage()
        storage.save_task("t1", "Task 1", workflow="hiring", budget_usd=1.0)
        storage.save_task("t2", "Task 2", workflow="hiring", budget_usd=2.0)
        storage.save_task("t3", "Task 3", workflow="hiring", budget_usd=3.0)

        storage.update_task_status("t1", "completed")
        storage.update_task_status("t2", "running")

        all_tasks = storage.list_tasks()
        assert len(all_tasks) == 3

        completed = storage.list_tasks(status="completed")
        assert len(completed) == 1
        assert completed[0]["task_id"] == "t1"


# ===================================================================
# 2. Escrow Double-Operation Edge Cases
# ===================================================================


class TestEscrowEdgeCases:
    """Verify escrow cannot be double-released, double-refunded, etc."""

    def test_double_release_returns_none(self):
        """Releasing an already-released escrow returns None."""
        escrow = AgentEscrow()
        entry = escrow.hold_payment("ceo", "agent1", 1.0, "t1")
        first = escrow.release_on_completion(entry.escrow_id)
        assert first is not None
        assert first.status == "released"

        second = escrow.release_on_completion(entry.escrow_id)
        assert second is None  # Already released

    def test_double_refund_returns_none(self):
        """Refunding an already-refunded escrow returns None."""
        escrow = AgentEscrow()
        entry = escrow.hold_payment("ceo", "agent1", 1.0, "t1")
        first = escrow.refund_on_failure(entry.escrow_id)
        assert first is not None
        assert first.status == "refunded"

        second = escrow.refund_on_failure(entry.escrow_id)
        assert second is None

    def test_release_after_refund_returns_none(self):
        """Cannot release an escrow that was already refunded."""
        escrow = AgentEscrow()
        entry = escrow.hold_payment("ceo", "agent1", 1.0, "t1")
        escrow.refund_on_failure(entry.escrow_id)

        result = escrow.release_on_completion(entry.escrow_id)
        assert result is None

    def test_refund_after_release_returns_none(self):
        """Cannot refund an escrow that was already released."""
        escrow = AgentEscrow()
        entry = escrow.hold_payment("ceo", "agent1", 1.0, "t1")
        escrow.release_on_completion(entry.escrow_id)

        result = escrow.refund_on_failure(entry.escrow_id)
        assert result is None

    def test_nonexistent_escrow_release_returns_none(self):
        """Releasing a nonexistent escrow returns None."""
        escrow = AgentEscrow()
        assert escrow.release_on_completion("fake-id") is None

    def test_nonexistent_escrow_refund_returns_none(self):
        """Refunding a nonexistent escrow returns None."""
        escrow = AgentEscrow()
        assert escrow.refund_on_failure("fake-id") is None

    def test_get_entries_for_task_returns_correct_entries(self):
        """get_entries_for_task returns only entries for that task."""
        escrow = AgentEscrow()
        escrow.hold_payment("ceo", "a1", 1.0, "task-a")
        escrow.hold_payment("ceo", "a2", 2.0, "task-b")
        escrow.hold_payment("ceo", "a3", 3.0, "task-a")

        entries = escrow.get_entries_for_task("task-a")
        assert len(entries) == 2
        assert all(e.task_id == "task-a" for e in entries)

    def test_clear_removes_all_entries(self):
        """clear() removes all escrow entries."""
        escrow = AgentEscrow()
        escrow.hold_payment("ceo", "a1", 1.0, "t1")
        escrow.hold_payment("ceo", "a2", 2.0, "t2")
        assert len(escrow.list_all()) == 2

        escrow.clear()
        assert len(escrow.list_all()) == 0
        assert escrow.total_held() == 0.0

    def test_total_released_accurate_after_mixed_operations(self):
        """total_released accurately reflects mixed release/refund."""
        escrow = AgentEscrow()
        e1 = escrow.hold_payment("ceo", "a1", 1.0, "t1")
        e2 = escrow.hold_payment("ceo", "a2", 2.0, "t2")
        e3 = escrow.hold_payment("ceo", "a3", 3.0, "t3")

        escrow.release_on_completion(e1.escrow_id)
        escrow.refund_on_failure(e2.escrow_id)
        escrow.release_on_completion(e3.escrow_id)

        assert escrow.total_released() == pytest.approx(4.0)
        assert escrow.total_held() == 0.0


# ===================================================================
# 3. Agent Timeout Simulation
# ===================================================================


class TestAgentTimeoutHandling:
    """Simulate agent timeouts and verify pipeline handles them."""

    def test_timeout_agent_triggers_refund(self):
        """When an agent times out, escrow should be refunded."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("SlowAgent", ["code"], price=0.10, agent_id="slow-1"))
        escrow = AgentEscrow()
        budget = BudgetTracker(total_budget=10.0)

        class TimingOutManager(HiringManager):
            def assign(self, agent, request):
                return {
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "task_id": request.task_id,
                    "status": "timeout",
                    "output": "",
                    "error": "Agent did not respond within deadline",
                }

        manager = TimingOutManager(registry=reg, escrow=escrow, budget_tracker=budget)
        result = manager.hire(HireRequest(required_skills=["code"], budget=5.0))

        assert result.status == "failed"
        assert result.escrow_id != ""

        entry = escrow.get_entry(result.escrow_id)
        assert entry.status == "refunded"

    def test_error_status_triggers_refund(self):
        """Agent returning error status triggers escrow refund."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("ErrorAgent", ["code"], price=0.05, agent_id="err-1"))
        escrow = AgentEscrow()
        budget = BudgetTracker(total_budget=10.0)

        class ErrorManager(HiringManager):
            def assign(self, agent, request):
                return {
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "task_id": request.task_id,
                    "status": "error",
                    "output": "",
                    "error": "Internal agent error",
                }

        manager = ErrorManager(registry=reg, escrow=escrow, budget_tracker=budget)
        result = manager.hire(HireRequest(required_skills=["code"], budget=5.0))

        assert result.status == "failed"
        entry = escrow.get_entry(result.escrow_id)
        assert entry.status == "refunded"


# ===================================================================
# 4. Concurrent Hiring Scenarios
# ===================================================================


class TestConcurrentHiringScenarios:
    """Multiple hires competing for budget and agents."""

    def test_rapid_sequential_hires_deplete_budget(self):
        """Many rapid hires deplete the budget correctly."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Worker", ["code"], price=0.25, agent_id="w1"))
        budget = BudgetTracker(total_budget=1.0)
        manager = HiringManager(registry=reg, budget_tracker=budget)

        results = []
        for i in range(8):
            r = manager.hire(HireRequest(required_skills=["code"], budget=1.0, description=f"Task {i}"))
            results.append(r)

        completed = [r for r in results if r.status == "completed"]
        failed = [r for r in results if r.status == "budget_exceeded"]

        assert len(completed) == 4  # 1.0 / 0.25 = 4 tasks
        assert len(failed) == 4
        assert budget.remaining == pytest.approx(0.0)

    def test_mixed_skill_hires_use_correct_agents(self):
        """Different skill requirements route to different agents."""
        manager, reg, escrow, bt = _create_pipeline(budget=100.0)

        skills_to_test = [
            (["code", "python"], "Coder"),
            (["design", "ui"], "Designer"),
            (["research", "analysis"], "Researcher"),
            (["deployment", "docker"], "DevOps"),
            (["writing", "docs"], "Writer"),
        ]

        for required_skills, expected_agent in skills_to_test:
            result = manager.hire(HireRequest(required_skills=required_skills, budget=10.0))
            assert result.agent_name == expected_agent, f"Expected {expected_agent} for skills {required_skills}"
            assert result.status == "completed"

    def test_hire_history_tracks_all_attempts(self):
        """hire_history includes both successes and failures."""
        manager, _, _, _ = _create_pipeline(budget=100.0)

        manager.hire(HireRequest(required_skills=["code"], budget=5.0))
        manager.hire(HireRequest(required_skills=["quantum"], budget=5.0))  # No match
        manager.hire(HireRequest(required_skills=["design"], budget=5.0))

        history = manager.hire_history
        assert len(history) == 3
        assert history[0].status == "completed"
        assert history[1].status == "no_agents"
        assert history[2].status == "completed"

    def test_identical_skill_agents_ranked_by_score(self):
        """When multiple agents share skills, ranking picks highest score."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Junior", ["code"], price=0.01, rating=3.0, agent_id="j1"))
        reg.register_agent(_make_listing("Senior", ["code", "python", "testing"], price=0.05, rating=4.8, agent_id="s1"))

        manager = HiringManager(registry=reg, budget_tracker=BudgetTracker(100.0))
        result = manager.hire(HireRequest(required_skills=["code", "python"], budget=10.0))

        # Senior has higher skill overlap (2/2 vs 1/2) so should be selected
        assert result.agent_name == "Senior"


# ===================================================================
# 5. HITL Integration with Hiring Pipeline
# ===================================================================


class TestHITLIntegration:
    """Human-in-the-Loop approval gate interacts with hiring pipeline."""

    def test_low_cost_auto_approved(self):
        """Low-cost actions are auto-approved without blocking."""
        gate = ApprovalGate(cost_threshold=10.0)
        req_id, requires_wait = gate.process_action(
            action="marketplace_hire",
            cost_usdc=0.50,
            description="Cheap hire",
        )
        assert requires_wait is False
        status = gate.check_approval(req_id)
        assert status == ApprovalStatus.AUTO_APPROVED

    def test_high_cost_requires_approval(self):
        """High-cost actions require human approval."""
        gate = ApprovalGate(cost_threshold=1.0)
        req_id, requires_wait = gate.process_action(
            action="marketplace_hire",
            cost_usdc=50.0,
            description="Expensive hire",
        )
        assert requires_wait is True
        status = gate.check_approval(req_id)
        assert status == ApprovalStatus.PENDING

    def test_approve_then_proceed(self):
        """Approved action transitions correctly."""
        gate = ApprovalGate(cost_threshold=1.0)
        req_id, _ = gate.process_action(
            action="hire",
            cost_usdc=5.0,
        )

        gate.approve(req_id, reviewer="sean", reason="Looks good")
        status = gate.check_approval(req_id)
        assert status == ApprovalStatus.APPROVED

        req = gate.get_request(req_id)
        assert req.reviewer == "sean"
        assert req.review_reason == "Looks good"
        assert req.reviewed_at > 0

    def test_reject_blocks_action(self):
        """Rejected action stays rejected."""
        gate = ApprovalGate(cost_threshold=1.0)
        req_id, _ = gate.process_action(
            action="hire",
            cost_usdc=5.0,
        )

        gate.reject(req_id, reviewer="sean", reason="Too expensive")
        status = gate.check_approval(req_id)
        assert status == ApprovalStatus.REJECTED

    def test_expired_request(self):
        """Expired requests transition to EXPIRED status."""
        gate = ApprovalGate(cost_threshold=1.0, timeout_seconds=0)
        req_id, _ = gate.process_action(
            action="hire",
            cost_usdc=5.0,
        )
        # Timeout is 0 seconds, so it's already expired
        time.sleep(0.01)
        status = gate.check_approval(req_id)
        assert status == ApprovalStatus.EXPIRED

    def test_cannot_approve_already_rejected(self):
        """Cannot approve a rejected request."""
        gate = ApprovalGate(cost_threshold=1.0)
        req_id, _ = gate.process_action(action="hire", cost_usdc=5.0)
        gate.reject(req_id)

        with pytest.raises(ValueError, match="Cannot approve"):
            gate.approve(req_id)

    def test_cannot_reject_already_approved(self):
        """Cannot reject an approved request."""
        gate = ApprovalGate(cost_threshold=1.0)
        req_id, _ = gate.process_action(action="hire", cost_usdc=5.0)
        gate.approve(req_id)

        with pytest.raises(ValueError, match="Cannot reject"):
            gate.reject(req_id)

    def test_list_pending_excludes_resolved(self):
        """list_pending only shows unresolved requests."""
        gate = ApprovalGate(cost_threshold=0.0)  # All require approval
        r1, _ = gate.process_action(action="hire", cost_usdc=1.0)
        r2, _ = gate.process_action(action="hire", cost_usdc=2.0)
        r3, _ = gate.process_action(action="hire", cost_usdc=3.0)

        gate.approve(r1)
        gate.reject(r2)

        pending = gate.list_pending()
        assert len(pending) == 1
        assert pending[0].request_id == r3

    def test_stats_track_all_outcomes(self):
        """Stats correctly count all outcome types."""
        gate = ApprovalGate(cost_threshold=5.0)

        # 2 auto-approved (below threshold)
        gate.process_action(action="a", cost_usdc=1.0)
        gate.process_action(action="b", cost_usdc=2.0)

        # 1 manually approved
        r3, _ = gate.process_action(action="c", cost_usdc=10.0)
        gate.approve(r3)

        # 1 rejected
        r4, _ = gate.process_action(action="d", cost_usdc=20.0)
        gate.reject(r4)

        stats = gate.get_stats()
        assert stats["total_requests"] == 4
        assert stats["auto_approved"] == 2
        assert stats["approved"] == 1
        assert stats["rejected"] == 1
        assert stats["pending"] == 0

    def test_unknown_request_raises(self):
        """Checking unknown request raises ValueError."""
        gate = ApprovalGate()
        with pytest.raises(ValueError, match="Unknown"):
            gate.check_approval("nonexistent-id")

    def test_clear_resets_state(self):
        """clear() resets all state."""
        gate = ApprovalGate(cost_threshold=0.0)
        gate.process_action(action="a", cost_usdc=1.0)
        gate.process_action(action="b", cost_usdc=2.0)

        gate.clear()
        assert len(gate.list_all()) == 0
        stats = gate.get_stats()
        assert stats["total_requests"] == 0


# ===================================================================
# 6. Payment Manager Edge Cases
# ===================================================================


class TestPaymentManagerEdgeCases:
    """Edge cases for PaymentManager: balance, debit, credit."""

    def test_credit_then_debit(self):
        """Credit and debit operations work correctly."""
        pm = PaymentManager()
        pm.credit("agent1", 5.0)
        assert pm.get_balance("agent1") == 5.0

        ok = pm.debit("agent1", 3.0)
        assert ok is True
        assert pm.get_balance("agent1") == pytest.approx(2.0)

    def test_debit_insufficient_balance_returns_false(self):
        """Debit with insufficient balance returns False."""
        pm = PaymentManager()
        pm.credit("agent1", 1.0)
        ok = pm.debit("agent1", 5.0)
        assert ok is False
        assert pm.get_balance("agent1") == 1.0  # Unchanged

    def test_debit_zero_balance_returns_false(self):
        """Debit from zero balance returns False."""
        pm = PaymentManager()
        ok = pm.debit("agent1", 0.01)
        assert ok is False

    def test_get_all_balances(self):
        """get_all_balances returns all credited agents."""
        pm = PaymentManager()
        pm.credit("a1", 1.0)
        pm.credit("a2", 2.0)
        pm.credit("a3", 3.0)

        balances = pm.get_all_balances()
        assert len(balances) == 3
        assert balances["a1"] == 1.0
        assert balances["a2"] == 2.0
        assert balances["a3"] == 3.0

    def test_multiple_credits_accumulate(self):
        """Multiple credits to the same agent accumulate."""
        pm = PaymentManager()
        pm.credit("agent1", 1.0)
        pm.credit("agent1", 2.5)
        pm.credit("agent1", 0.5)
        assert pm.get_balance("agent1") == pytest.approx(4.0)

    def test_release_escrow_nonexistent_returns_none(self):
        """Releasing nonexistent escrow returns None."""
        pm = PaymentManager()
        assert pm.release_escrow("nonexistent") is None

    def test_refund_escrow_nonexistent_returns_none(self):
        """Refunding nonexistent escrow returns None."""
        pm = PaymentManager()
        assert pm.refund_escrow("nonexistent") is None

    def test_full_payment_lifecycle_with_ledger(self):
        """Complete lifecycle: request → hold → release → verify ledger."""
        pm = PaymentManager(
            gate=X402PaymentGate(PaymentConfig(price=0.10, pay_to="0xSeller")),
        )

        # Request payment
        resp = pm.create_payment_request("/test", 0.10, "0xSeller", "Test payment")
        assert resp["error"] == "Payment Required"

        # Hold escrow
        entry = pm.hold_escrow("buyer", "seller", 0.10, "task-1")
        assert entry.status == "held"

        # Release escrow
        released = pm.release_escrow(entry.escrow_id)
        assert released.status == "released"
        assert pm.get_balance("seller") == pytest.approx(0.10)

        # Verify full ledger trail
        all_entries = pm.ledger.get_all()
        types = [e.event_type for e in all_entries]
        assert "payment_request" in types
        assert "escrow_hold" in types
        assert "escrow_release" in types

    def test_refund_credits_payer_not_payee(self):
        """Refund should credit the payer, not the payee."""
        pm = PaymentManager()
        entry = pm.hold_escrow("buyer", "seller", 5.0, "task-refund")
        pm.refund_escrow(entry.escrow_id)

        assert pm.get_balance("buyer") == pytest.approx(5.0)
        assert pm.get_balance("seller") == 0.0


# ===================================================================
# 7. X402 Payment Gate Edge Cases
# ===================================================================


class TestX402PaymentGateEdgeCases:
    """Edge cases for the x402 payment gate."""

    def test_price_override_in_402_response(self):
        """Price override should affect the 402 response."""
        gate = X402PaymentGate(PaymentConfig(price=1.0, pay_to="0xA"))
        resp = gate.create_402_response(resource="/api/test", price_override=5.0)

        assert resp["accepts"][0]["maxAmountRequired"] == "5000000"  # 5.0 * 1M

    def test_zero_price_accepted(self):
        """Zero price generates valid 402 response (free with protocol)."""
        gate = X402PaymentGate(PaymentConfig(price=0.0, pay_to="0xA"))
        resp = gate.create_402_response("/free")
        assert resp["accepts"][0]["maxAmountRequired"] == "0"

    def test_amount_insufficient_verification_fails(self):
        """Payment less than required price fails verification."""
        gate = X402PaymentGate(PaymentConfig(price=1.0, pay_to="0xSeller"))
        proof = PaymentProof(payer="0xBuyer", payee="0xSeller", amount=0.50)
        assert gate.verify_payment(proof) is False

    def test_payment_history_filtered_by_payer(self):
        """Payment history filters by payer correctly."""
        gate = X402PaymentGate(PaymentConfig(price=0.0, pay_to=""))
        gate.verify_payment(PaymentProof(payer="alice", payee="bob", amount=1.0))
        gate.verify_payment(PaymentProof(payer="charlie", payee="bob", amount=2.0))
        gate.verify_payment(PaymentProof(payer="alice", payee="dave", amount=3.0))

        alice_payments = gate.payment_history(payer="alice")
        assert len(alice_payments) == 2

        all_payments = gate.payment_history()
        assert len(all_payments) == 3

    def test_total_collected(self):
        """total_collected sums verified payments."""
        gate = X402PaymentGate(PaymentConfig(price=0.0, pay_to=""))
        gate.verify_payment(PaymentProof(payer="a", payee="b", amount=1.0))
        gate.verify_payment(PaymentProof(payer="a", payee="b", amount=2.5))

        assert gate.total_collected() == pytest.approx(3.5)

    def test_is_paid_resource(self):
        """Record and check paid resources."""
        gate = X402PaymentGate()
        proof = PaymentProof(payer="a", payee="b", amount=1.0)

        assert gate.is_paid("/api/resource") is False
        gate.record_verified_payment("/api/resource", proof)
        assert gate.is_paid("/api/resource") is True
        assert gate.is_paid("/api/other") is False

    def test_accepts_entry_has_facilitator_url(self):
        """Accepts entry includes the facilitator URL."""
        config = PaymentConfig(
            price=1.0,
            pay_to="0xAddr",
            facilitator_url="https://custom-facilitator.example.com",
        )
        entry = config.to_accepts_entry("/test", "Test")
        assert entry["extra"]["facilitatorUrl"] == "https://custom-facilitator.example.com"


# ===================================================================
# 8. BudgetTracker Edge Cases
# ===================================================================


class TestBudgetTrackerEdgeCases:
    """Edge cases for BudgetTracker."""

    def test_exact_budget_spend(self):
        """Spending exactly the remaining budget succeeds."""
        bt = BudgetTracker(total_budget=1.0)
        assert bt.spend("t1", 1.0) is True
        assert bt.remaining == 0.0
        assert bt.can_afford(0.01) is False

    def test_over_budget_spend_fails(self):
        """Spending more than remaining budget fails."""
        bt = BudgetTracker(total_budget=1.0)
        assert bt.spend("t1", 1.01) is False
        assert bt.remaining == 1.0

    def test_multiple_spends_same_task_accumulate(self):
        """Multiple spends on the same task_id accumulate."""
        bt = BudgetTracker(total_budget=10.0)
        bt.spend("t1", 1.0)
        bt.spend("t1", 2.0)
        assert bt.get_spending("t1") == 3.0
        assert bt.total_spent == 3.0

    def test_reset_clears_all_spending(self):
        """reset() clears all spending data."""
        bt = BudgetTracker(total_budget=10.0)
        bt.spend("t1", 5.0)
        bt.reset()
        assert bt.total_spent == 0.0
        assert bt.remaining == 10.0

    def test_spending_report_empty(self):
        """Spending report with no spending."""
        bt = BudgetTracker(total_budget=50.0)
        report = bt.spending_report()
        assert report["total_budget"] == 50.0
        assert report["total_spent"] == 0.0
        assert report["remaining"] == 50.0
        assert len(report["tasks"]) == 0


# ===================================================================
# 9. Marketplace Registry Advanced Cases
# ===================================================================


class TestMarketplaceRegistryAdvanced:
    """Advanced marketplace registry scenarios."""

    def test_unregister_agent(self):
        """Unregistering removes agent from registry."""
        reg = MarketplaceRegistry()
        listing = _make_listing("Agent1", ["code"], agent_id="a1")
        reg.register_agent(listing)
        assert reg.count() == 1

        assert reg.unregister_agent("a1") is True
        assert reg.count() == 0
        assert reg.get_agent("a1") is None

    def test_unregister_nonexistent_returns_false(self):
        """Unregistering nonexistent agent returns False."""
        reg = MarketplaceRegistry()
        assert reg.unregister_agent("ghost") is False

    def test_get_agent_by_name(self):
        """Can find agent by name."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("SpecialAgent", ["spy"], agent_id="spy-1"))

        found = reg.get_agent_by_name("SpecialAgent")
        assert found is not None
        assert found.agent_id == "spy-1"

    def test_get_agent_by_name_not_found(self):
        """get_agent_by_name returns None for unknown name."""
        reg = MarketplaceRegistry()
        assert reg.get_agent_by_name("Ghost") is None

    def test_discover_by_description(self):
        """discover_agents matches against description too."""
        reg = MarketplaceRegistry()
        listing = AgentListing(
            name="GenericAgent",
            description="Specializes in quantum computing simulations",
            skills=["code"],
            price_per_unit=0.01,
        )
        reg.register_agent(listing)

        results = reg.discover_agents("quantum")
        assert len(results) == 1
        assert results[0].name == "GenericAgent"

    def test_discover_with_price_filter(self):
        """discover_agents respects max price filter."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Cheap", ["code"], price=0.01))
        reg.register_agent(_make_listing("Mid", ["code"], price=0.50))
        reg.register_agent(_make_listing("Expensive", ["code"], price=10.0))

        results = reg.discover_agents("code", max_price=1.0)
        names = [a.name for a in results]
        assert "Cheap" in names
        assert "Mid" in names
        assert "Expensive" not in names

    def test_sort_by_price_ascending(self):
        """sort_by_price returns in ascending order."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Mid", ["code"], price=0.50))
        reg.register_agent(_make_listing("Low", ["code"], price=0.01))
        reg.register_agent(_make_listing("High", ["code"], price=5.0))

        sorted_agents = reg.sort_by_price(ascending=True)
        prices = [a.price_per_unit for a in sorted_agents]
        assert prices == sorted(prices)

    def test_agent_listing_properties(self):
        """AgentListing computed properties work correctly."""
        listing = AgentListing(
            name="Test",
            pricing_model="per-task",
            price_per_unit=0.05,
            total_jobs=10,
            completed_jobs=8,
            failed_jobs=2,
        )

        assert listing.price_display == "$0.0500/task"
        assert listing.completion_rate == pytest.approx(0.8)
        assert listing.matches_skill("test") is False
        assert listing.to_dict()["completion_rate"] == pytest.approx(0.8)

    def test_per_token_pricing_display(self):
        """per-token pricing model displays correctly."""
        listing = AgentListing(pricing_model="per-token", price_per_unit=0.0030)
        assert listing.price_display == "$0.0030/1K tokens"

    def test_zero_jobs_completion_rate(self):
        """Zero jobs gives 0.0 completion rate (no division by zero)."""
        listing = AgentListing(total_jobs=0)
        assert listing.completion_rate == 0.0


# ===================================================================
# 10. SkillMatcher Edge Cases
# ===================================================================


class TestSkillMatcherEdgeCases:
    """Edge cases for skill matching algorithm."""

    def test_empty_skills_returns_all(self):
        """Empty required_skills returns all agents sorted by rating."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("A", ["code"], rating=3.0))
        reg.register_agent(_make_listing("B", ["design"], rating=5.0))

        matcher = SkillMatcher(reg)
        results = matcher.match([])
        assert len(results) == 2
        # B should be first (higher rating)
        assert results[0][0].name == "B"

    def test_no_match_returns_empty(self):
        """No matching skills returns empty list."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Coder", ["python"]))

        matcher = SkillMatcher(reg)
        results = matcher.match(["quantum-computing"])
        assert len(results) == 0

    def test_best_match_returns_single(self):
        """best_match returns a single result."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("A", ["code"], rating=4.0))
        reg.register_agent(_make_listing("B", ["code"], rating=3.0))

        matcher = SkillMatcher(reg)
        result = matcher.best_match(["code"])
        assert result is not None
        assert result[0].name == "A"

    def test_best_match_no_match_returns_none(self):
        """best_match returns None when no agents match."""
        reg = MarketplaceRegistry()
        matcher = SkillMatcher(reg)
        assert matcher.best_match(["quantum"]) is None

    def test_top_n_limits_results(self):
        """top_n parameter limits number of results."""
        reg = MarketplaceRegistry()
        for i in range(10):
            reg.register_agent(_make_listing(f"Agent{i}", ["code"], rating=float(i), agent_id=f"a{i}"))

        matcher = SkillMatcher(reg)
        results = matcher.match(["code"], top_n=3)
        assert len(results) == 3

    def test_case_insensitive_matching(self):
        """Skill matching is case-insensitive."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Coder", ["Python", "JavaScript"]))

        matcher = SkillMatcher(reg)
        results = matcher.match(["python"])
        assert len(results) == 1

    def test_partial_skill_overlap_included(self):
        """Agents with partial skill overlap are included."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Full", ["python", "testing", "deploy"], rating=4.0))
        reg.register_agent(_make_listing("Half", ["python"], rating=4.0))

        matcher = SkillMatcher(reg)
        results = matcher.match(["python", "testing", "deploy"])
        assert len(results) == 2

        names = {r[0].name for r in results}
        assert "Full" in names
        assert "Half" in names


# ===================================================================
# 11. Pipeline Audit Trail Consistency
# ===================================================================


class TestAuditTrailConsistency:
    """Verify that the audit trail is complete and consistent after pipeline runs."""

    def test_successful_hire_creates_complete_audit_trail(self):
        """A successful hire should produce escrow entries, budget deduction, and reputation update."""
        manager, reg, escrow, bt = _create_pipeline(budget=10.0)

        result = manager.hire(HireRequest(
            required_skills=["code"],
            budget=5.0,
            description="Audit trail test",
        ))

        # Pipeline completed
        assert result.status == "completed"

        # Escrow released
        entry = escrow.get_entry(result.escrow_id)
        assert entry.status == "released"
        assert entry.resolved_at is not None
        assert entry.amount == result.agreed_price
        assert entry.task_id == result.task_id

        # Budget deducted
        assert bt.get_spending(result.task_id) == result.agreed_price

        # Agent reputation updated
        agent = reg.get_agent(result.agent_id)
        assert agent.total_jobs >= 1

        # Hire history populated
        assert len(manager.hire_history) == 1
        assert manager.hire_history[0].task_id == result.task_id

    def test_failed_hire_has_no_budget_deduction(self):
        """A failed hire (no agents) should not deduct from budget."""
        reg = MarketplaceRegistry()  # Empty
        bt = BudgetTracker(total_budget=10.0)
        manager = HiringManager(registry=reg, budget_tracker=bt)

        result = manager.hire(HireRequest(
            required_skills=["quantum"],
            budget=5.0,
        ))

        assert result.status == "no_agents"
        assert bt.total_spent == 0.0
        assert bt.remaining == 10.0

    def test_pipeline_idempotent_fresh_state(self):
        """Each pipeline run starts with independent state."""
        for _ in range(3):
            manager, reg, escrow, bt = _create_pipeline(budget=10.0)
            result = manager.hire(HireRequest(required_skills=["code"], budget=5.0))
            assert result.status == "completed"
            assert len(escrow.list_all()) == 1
            assert len(manager.hire_history) == 1


# ===================================================================
# 12. HireRequest / HireResult Data Integrity
# ===================================================================


class TestDataIntegrity:
    """Verify data integrity of HireRequest and HireResult dataclasses."""

    def test_hire_request_generates_unique_ids(self):
        """Each HireRequest gets a unique task_id."""
        r1 = HireRequest(description="Task 1")
        r2 = HireRequest(description="Task 2")
        assert r1.task_id != r2.task_id
        assert r1.task_id.startswith("hire_")

    def test_hire_request_defaults(self):
        """HireRequest has sensible defaults."""
        r = HireRequest()
        assert r.description == ""
        assert r.required_skills == []
        assert r.budget == 0.0
        assert r.requester == "ceo"

    def test_hire_result_defaults(self):
        """HireResult has sensible defaults."""
        r = HireResult()
        assert r.status == "pending"
        assert r.agent_id == ""
        assert r.agent_name == ""
        assert r.escrow_id == ""
        assert r.agreed_price == 0.0
        assert r.task_result is None
        assert r.error == ""
        assert r.elapsed_s == 0.0
        assert r.budget_remaining == 0.0

    def test_escrow_entry_defaults(self):
        """EscrowEntry has correct defaults."""
        entry = EscrowEntry()
        assert entry.escrow_id.startswith("escrow_")
        assert entry.status == "held"
        assert entry.resolved_at is None
        assert entry.payer == ""
        assert entry.payee == ""
        assert entry.amount == 0.0

    def test_payment_proof_defaults(self):
        """PaymentProof has correct defaults."""
        proof = PaymentProof()
        assert proof.payment_id.startswith("pay_")
        assert proof.network == "eip155:8453"
        assert proof.verified is False
        assert proof.timestamp > 0


# ===================================================================
# 13. API Endpoint Method Validation
# ===================================================================


@pytest.fixture
def api_client():
    """Create a test client with marketplace routes and fresh state."""
    import src.api.marketplace_routes as routes_mod

    fresh_marketplace = MarketplaceRegistry()
    fresh_marketplace.register_agent(_make_listing("APICoder", ["code", "python"], 0.01, 4.5, "api-coder-001"))

    fresh_escrow = AgentEscrow()
    fresh_budget = BudgetTracker(total_budget=100.0)
    fresh_hiring = HiringManager(
        registry=fresh_marketplace,
        escrow=fresh_escrow,
        budget_tracker=fresh_budget,
    )

    old_marketplace = routes_mod.marketplace
    old_escrow = routes_mod._escrow
    old_budget = routes_mod._budget
    old_hiring = routes_mod._hiring_manager

    routes_mod.marketplace = fresh_marketplace
    routes_mod._escrow = fresh_escrow
    routes_mod._budget = fresh_budget
    routes_mod._hiring_manager = fresh_hiring

    app = FastAPI()
    app.include_router(routes_mod.router)
    client = TestClient(app)
    yield client

    routes_mod.marketplace = old_marketplace
    routes_mod._escrow = old_escrow
    routes_mod._budget = old_budget
    routes_mod._hiring_manager = old_hiring


class TestAPIMethodValidation:
    """Verify API endpoints reject wrong HTTP methods."""

    def test_agents_get_accepted(self, api_client):
        resp = api_client.get("/marketplace/agents")
        assert resp.status_code == 200

    def test_agents_delete_rejected(self, api_client):
        resp = api_client.delete("/marketplace/agents")
        assert resp.status_code == 405

    def test_agents_put_rejected(self, api_client):
        resp = api_client.put("/marketplace/agents", json={})
        assert resp.status_code == 405

    def test_hire_get_rejected(self, api_client):
        resp = api_client.get("/marketplace/hire")
        assert resp.status_code == 405

    def test_budget_post_rejected(self, api_client):
        resp = api_client.post("/marketplace/budget", json={})
        assert resp.status_code == 405

    def test_ledger_post_rejected(self, api_client):
        resp = api_client.post("/payments/ledger", json={})
        assert resp.status_code == 405


# ===================================================================
# 14. API Content Validation
# ===================================================================


class TestAPIContentValidation:
    """Verify API response content is well-formed."""

    def test_hire_response_contains_all_fields(self, api_client):
        """Hire response has all expected fields."""
        resp = api_client.post("/marketplace/hire", json={
            "description": "Content validation test",
            "required_skills": ["code"],
            "budget": 5.0,
        })
        assert resp.status_code == 200
        data = resp.json()

        required_fields = ["task_id", "status", "agent_id", "agent_name",
                           "agreed_price", "escrow_id", "budget_remaining"]
        for f in required_fields:
            assert f in data, f"Missing field: {f}"

    def test_402_response_matches_x402_spec(self, api_client):
        """Payment request returns valid x402 V2 response."""
        resp = api_client.post("/payments/request", json={
            "resource": "/api/test",
            "amount": 1.0,
            "payee": "agent1",
        })
        assert resp.status_code == 402
        data = resp.json()

        assert data["error"] == "Payment Required"
        assert len(data["accepts"]) >= 1

        accepts = data["accepts"][0]
        assert "scheme" in accepts
        assert "network" in accepts
        assert "maxAmountRequired" in accepts
        assert "payTo" in accepts
        assert "extra" in accepts
        assert "facilitatorUrl" in accepts["extra"]

    def test_hire_with_long_description(self, api_client):
        """Hire accepts descriptions up to 2000 chars."""
        long_desc = "A" * 2000
        resp = api_client.post("/marketplace/hire", json={
            "description": long_desc,
            "required_skills": ["code"],
            "budget": 1.0,
        })
        assert resp.status_code == 200

    def test_hire_too_long_description_returns_422(self, api_client):
        """Hire rejects descriptions over 2000 chars."""
        too_long = "A" * 2001
        resp = api_client.post("/marketplace/hire", json={
            "description": too_long,
            "required_skills": ["code"],
            "budget": 1.0,
        })
        assert resp.status_code == 422

    def test_register_agent_too_long_name_returns_422(self, api_client):
        """Agent registration rejects names over 200 chars."""
        resp = api_client.post("/marketplace/agents", json={
            "name": "A" * 201,
            "skills": ["code"],
        })
        assert resp.status_code == 422

    def test_agent_with_negative_price_returns_422(self, api_client):
        """Agent registration rejects negative price."""
        resp = api_client.post("/marketplace/agents", json={
            "name": "BadAgent",
            "skills": ["code"],
            "price_per_unit": -1.0,
        })
        assert resp.status_code == 422


# ===================================================================
# 15. Full Demo Pipeline (API -> Hire -> Budget -> Ledger)
# ===================================================================


class TestFullDemoPipeline:
    """End-to-end demo pipeline that judges would evaluate."""

    def test_demo_flow_register_discover_hire_verify(self, api_client):
        """Complete demo flow via REST API."""
        # 1. Register a specialized agent
        reg_resp = api_client.post("/marketplace/agents", json={
            "name": "DemoAgent",
            "skills": ["demo", "testing", "integration"],
            "price_per_unit": 0.02,
        })
        assert reg_resp.status_code == 201
        agent_id = reg_resp.json()["agent_id"]

        # 2. Discover agents with demo skills
        discover_resp = api_client.get("/marketplace/agents?skill=demo")
        assert discover_resp.status_code == 200
        agents = discover_resp.json()
        assert any(a["agent_id"] == agent_id for a in agents)

        # 3. Hire the agent
        hire_resp = api_client.post("/marketplace/hire", json={
            "description": "Run integration demo for judges",
            "required_skills": ["demo", "testing"],
            "budget": 1.0,
        })
        assert hire_resp.status_code == 200
        hire_data = hire_resp.json()
        assert hire_data["status"] == "completed"
        assert hire_data["agreed_price"] == 0.02
        task_id = hire_data["task_id"]

        # 4. Check hire status
        status_resp = api_client.get(f"/marketplace/hire/{task_id}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["escrow_status"] == "released"

        # 5. Verify budget was deducted
        budget_resp = api_client.get("/marketplace/budget")
        assert budget_resp.status_code == 200
        budget = budget_resp.json()
        assert budget["total_spent"] > 0

        # 6. Check jobs list
        jobs_resp = api_client.get("/marketplace/jobs")
        assert jobs_resp.status_code == 200
        jobs = jobs_resp.json()
        assert any(j["task_id"] == task_id for j in jobs)

    def test_demo_flow_payment_request_and_verify(self, api_client):
        """x402 payment flow: request → verify → check balance → ledger."""
        # 1. Create payment request
        req_resp = api_client.post("/payments/request", json={
            "resource": "/demo/test",
            "amount": 0.50,
            "payee": "demo-agent",
            "description": "Demo payment for judges",
        })
        assert req_resp.status_code == 402
        assert req_resp.headers.get("X-Payment") == "required"

        # 2. Verify payment (payee must match the gate's pay_to or gate has empty pay_to)
        # The API's _payment_manager has a gate configured with a specific pay_to address.
        # For the verify endpoint, the gate checks payee == pay_to. Since the API
        # singleton gate has a fixed address, we test the endpoint returns a valid response.
        verify_resp = api_client.post("/payments/verify", json={
            "payer": "0xJudge",
            "payee": "0xSomeAgent",
            "amount": 0.50,
        })
        assert verify_resp.status_code == 200
        data = verify_resp.json()
        assert "verified" in data
        assert "payment_id" in data

        # 3. Check balance endpoint works
        balance_resp = api_client.get("/payments/balance/demo-agent")
        assert balance_resp.status_code == 200

        # 4. Check ledger endpoint works
        ledger_resp = api_client.get("/payments/ledger")
        assert ledger_resp.status_code == 200

    def test_demo_flow_multiple_agents_different_skills(self, api_client):
        """Register multiple agents and hire the right one for each task."""
        # Register agents with different specialties
        for name, skills, price in [
            ("CodeExpert", ["code", "python", "api"], 0.03),
            ("UIExpert", ["design", "frontend", "css"], 0.04),
            ("DataExpert", ["data", "analysis", "ml"], 0.05),
        ]:
            resp = api_client.post("/marketplace/agents", json={
                "name": name,
                "skills": skills,
                "price_per_unit": price,
            })
            assert resp.status_code == 201

        # Hire for code task
        code_hire = api_client.post("/marketplace/hire", json={
            "description": "Build REST API",
            "required_skills": ["code", "api"],
            "budget": 1.0,
        })
        assert code_hire.status_code == 200
        assert code_hire.json()["agent_name"] == "CodeExpert"

        # Hire for design task
        design_hire = api_client.post("/marketplace/hire", json={
            "description": "Design landing page",
            "required_skills": ["design", "frontend"],
            "budget": 1.0,
        })
        assert design_hire.status_code == 200
        assert design_hire.json()["agent_name"] == "UIExpert"

        # Hire for data task
        data_hire = api_client.post("/marketplace/hire", json={
            "description": "Analyze user behavior",
            "required_skills": ["data", "analysis"],
            "budget": 1.0,
        })
        assert data_hire.status_code == 200
        assert data_hire.json()["agent_name"] == "DataExpert"

        # All 3 jobs appear in list
        jobs = api_client.get("/marketplace/jobs").json()
        assert len(jobs) == 3
        assert all(j["status"] == "completed" for j in jobs)

        # Budget reflects all 3 hires
        budget = api_client.get("/marketplace/budget").json()
        assert budget["total_spent"] == pytest.approx(0.03 + 0.04 + 0.05)
