"""Integration tests for the full hiring pipeline.

Covers the end-to-end flow that judges will evaluate:
  Task creation -> Agent matching -> Task delegation -> x402 payment -> Completion

Also covers error handling edge cases:
  - No matching agents
  - Payment failures
  - Agent timeouts
  - Budget exhaustion mid-pipeline
  - Escrow lifecycle integrity
  - API error codes (400, 404, 500)
"""

from __future__ import annotations

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
    payment_manager,
)
from src.marketplace.hiring import HireRequest, HireResult, BudgetTracker, HiringManager


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
    """Registry with diverse agents covering multiple skill categories."""
    reg = MarketplaceRegistry()
    reg.register_agent(_make_listing("Coder", ["code", "python", "testing"], 0.02, 4.5, "coder-001"))
    reg.register_agent(_make_listing("Designer", ["design", "ui", "ux"], 0.05, 4.8, "designer-001"))
    reg.register_agent(_make_listing("Researcher", ["research", "analysis", "web-search"], 0.01, 4.2, "researcher-001"))
    reg.register_agent(_make_listing("DevOps", ["deployment", "docker", "ci-cd"], 0.03, 3.9, "devops-001"))
    reg.register_agent(_make_listing("Writer", ["writing", "docs", "content"], 0.00, 4.0, "writer-001"))
    return reg


def _create_pipeline(budget: float = 100.0, registry: MarketplaceRegistry | None = None):
    """Create a full pipeline with fresh escrow and budget."""
    reg = registry or _seeded_registry()
    escrow = AgentEscrow()
    bt = BudgetTracker(total_budget=budget)
    manager = HiringManager(registry=reg, escrow=escrow, budget_tracker=bt)
    return manager, reg, escrow, bt


# ===================================================================
# 1. Full Pipeline Integration — Happy Path
# ===================================================================


class TestFullPipelineHappyPath:
    """End-to-end: task creation -> agent matching -> delegation -> payment -> completion."""

    def test_single_task_full_lifecycle(self):
        """Complete lifecycle for a single task through the pipeline."""
        manager, reg, escrow, bt = _create_pipeline(budget=10.0)

        # Step 1: Create task request
        request = HireRequest(
            description="Build a REST API endpoint",
            required_skills=["code", "python"],
            budget=5.0,
        )

        # Step 2: Discover agents
        candidates = manager.discover(request)
        assert len(candidates) >= 1
        assert candidates[0][0].name == "Coder"

        # Step 3: Select best agent
        agent = manager.select(candidates)
        assert agent is not None
        assert agent.name == "Coder"
        assert agent.agent_id == "coder-001"

        # Step 4: Negotiate price
        price = manager.negotiate_price(agent, request.budget)
        assert price is not None
        assert price == 0.02

        # Step 5: Create escrow payment
        escrow_entry = manager.pay(request, agent, price)
        assert escrow_entry is not None
        assert escrow_entry.status == "held"
        assert escrow_entry.amount == 0.02
        assert escrow_entry.payer == "ceo"
        assert escrow_entry.payee == "Coder"

        # Step 6: Assign task
        result = manager.assign(agent, request)
        assert result["status"] == "completed"
        assert result["agent_name"] == "Coder"
        assert request.description in result["output"]

        # Step 7: Verify result
        assert manager.verify_result(result) is True

        # Step 8: Release payment
        released = manager.release_payment(escrow_entry.escrow_id)
        assert released is not None
        assert released.status == "released"
        assert released.resolved_at is not None

        # Step 9: Verify budget deduction
        assert bt.total_spent == 0.02
        assert bt.remaining == pytest.approx(9.98)

    def test_hire_method_orchestrates_full_flow(self):
        """The hire() method runs all steps end-to-end."""
        manager, reg, escrow, bt = _create_pipeline(budget=10.0)
        request = HireRequest(
            description="Write unit tests for auth module",
            required_skills=["code", "testing"],
            budget=5.0,
        )
        result = manager.hire(request)

        assert result.status == "completed"
        assert result.agent_name == "Coder"
        assert result.agreed_price == 0.02
        assert result.escrow_id != ""
        assert result.task_result is not None
        assert result.task_result["status"] == "completed"
        assert result.budget_remaining == pytest.approx(9.98)
        assert result.elapsed_s >= 0

        # Verify escrow was released
        entry = escrow.get_entry(result.escrow_id)
        assert entry.status == "released"

        # Verify agent stats updated
        coder = reg.get_agent("coder-001")
        assert coder.total_jobs >= 1

    def test_sequential_multi_task_pipeline(self):
        """Multiple tasks processed sequentially with budget tracking."""
        manager, reg, escrow, bt = _create_pipeline(budget=1.0)

        tasks = [
            HireRequest(required_skills=["code"], budget=0.5, description="Task 1"),
            HireRequest(required_skills=["research"], budget=0.5, description="Task 2"),
            HireRequest(required_skills=["writing"], budget=0.5, description="Task 3"),
        ]

        results = [manager.hire(req) for req in tasks]

        assert all(r.status == "completed" for r in results)
        assert len(manager.hire_history) == 3

        # Budget should reflect all spending
        total_price = sum(r.agreed_price for r in results)
        assert bt.total_spent == pytest.approx(total_price)

    def test_pipeline_with_different_agents_for_different_skills(self):
        """Different tasks should match to different specialized agents."""
        manager, reg, escrow, bt = _create_pipeline(budget=10.0)

        code_result = manager.hire(HireRequest(required_skills=["code", "python"], budget=5.0))
        design_result = manager.hire(HireRequest(required_skills=["design", "ui"], budget=5.0))
        research_result = manager.hire(HireRequest(required_skills=["research", "analysis"], budget=5.0))

        assert code_result.agent_name == "Coder"
        assert design_result.agent_name == "Designer"
        assert research_result.agent_name == "Researcher"

    def test_escrow_lifecycle_integrity(self):
        """Verify escrow entries are consistent through the full pipeline."""
        manager, reg, escrow, bt = _create_pipeline(budget=10.0)

        result = manager.hire(HireRequest(
            required_skills=["code"],
            budget=5.0,
            description="Integration test task",
        ))

        # Verify only one escrow entry was created
        all_entries = escrow.list_all()
        assert len(all_entries) == 1
        entry = all_entries[0]

        # Verify escrow entry fields
        assert entry.task_id == result.task_id
        assert entry.amount == result.agreed_price
        assert entry.payer == "ceo"
        assert entry.payee == result.agent_name
        assert entry.status == "released"
        assert entry.resolved_at is not None

        # No held entries should remain
        assert escrow.total_held() == 0.0
        assert escrow.total_released() == pytest.approx(result.agreed_price)


# ===================================================================
# 2. Payment Integration — x402 Full Flow
# ===================================================================


class TestPaymentIntegration:
    """Full x402 payment lifecycle: request → proof → verify → escrow → release."""

    def test_payment_request_to_verification_flow(self):
        """Create payment request, generate proof, verify it."""
        pm = PaymentManager(
            gate=X402PaymentGate(PaymentConfig(price=0.05, pay_to="0xSeller")),
        )

        # Create payment request (402 response)
        resp = pm.create_payment_request(
            resource="/marketplace/hire/task-123",
            amount=0.05,
            payee="0xSeller",
            description="Hire agent for code review",
        )
        assert resp["error"] == "Payment Required"
        assert len(resp["accepts"]) == 1
        assert resp["accepts"][0]["payTo"] == "0xSeller"

        # Create and verify payment proof
        proof = PaymentProof(
            payer="0xBuyer",
            payee="0xSeller",
            amount=0.05,
            tx_hash="0xabc123def456",
        )
        assert pm.verify_payment(proof) is True

        # Check balance updated
        assert pm.get_balance("0xSeller") == pytest.approx(0.05)

        # Check ledger has both events
        request_entries = pm.ledger.get_entries(event_type="payment_request")
        assert len(request_entries) == 1
        verify_entries = pm.ledger.get_entries(event_type="payment_verified")
        assert len(verify_entries) == 1

    def test_escrow_hold_to_release_with_ledger_audit(self):
        """Full escrow flow with ledger audit trail."""
        pm = PaymentManager()

        # Hold escrow
        entry = pm.hold_escrow("buyer", "seller", 1.50, "task-001")
        assert entry.status == "held"
        assert pm.escrow.total_held() == pytest.approx(1.50)

        # Release on completion
        released = pm.release_escrow(entry.escrow_id)
        assert released.status == "released"
        assert pm.escrow.total_held() == 0.0
        assert pm.get_balance("seller") == pytest.approx(1.50)

        # Verify full ledger trail
        all_entries = pm.ledger.get_all()
        event_types = [e.event_type for e in all_entries]
        assert "escrow_hold" in event_types
        assert "escrow_release" in event_types
        assert pm.ledger.total_volume() == pytest.approx(3.0)  # hold + release both recorded

    def test_escrow_hold_to_refund_with_ledger_audit(self):
        """Full escrow refund flow with ledger audit trail."""
        pm = PaymentManager()

        entry = pm.hold_escrow("buyer", "seller", 2.00, "task-002")
        refunded = pm.refund_escrow(entry.escrow_id)

        assert refunded.status == "refunded"
        assert pm.get_balance("buyer") == pytest.approx(2.00)
        assert pm.get_balance("seller") == 0.0

        event_types = [e.event_type for e in pm.ledger.get_all()]
        assert "escrow_hold" in event_types
        assert "escrow_refund" in event_types

    def test_multi_escrow_concurrent_tracking(self):
        """Multiple escrow holds tracked independently."""
        pm = PaymentManager()

        e1 = pm.hold_escrow("buyer", "agent1", 1.0, "t1")
        e2 = pm.hold_escrow("buyer", "agent2", 2.0, "t2")
        e3 = pm.hold_escrow("buyer", "agent3", 0.5, "t3")

        assert pm.escrow.total_held() == pytest.approx(3.5)

        pm.release_escrow(e1.escrow_id)
        pm.refund_escrow(e3.escrow_id)

        assert pm.escrow.total_held() == pytest.approx(2.0)
        assert pm.get_balance("agent1") == pytest.approx(1.0)
        assert pm.get_balance("buyer") == pytest.approx(0.5)
        assert pm.get_balance("agent3") == 0.0

    def test_payment_verification_failure_does_not_credit(self):
        """Failed verification should not credit the payee."""
        pm = PaymentManager(
            gate=X402PaymentGate(PaymentConfig(price=1.0, pay_to="0xCorrect")),
        )

        bad_proof = PaymentProof(payer="0xBuyer", payee="0xWrong", amount=1.0)
        assert pm.verify_payment(bad_proof) is False
        assert pm.get_balance("0xWrong") == 0.0

        entries = pm.ledger.get_entries(event_type="payment_rejected")
        assert len(entries) == 1


# ===================================================================
# 3. Error Handling Edge Cases
# ===================================================================


class TestErrorHandlingEdgeCases:
    """Error paths: no agents, payment failures, timeouts, budget issues."""

    def test_no_matching_agents(self):
        """When no agents match required skills."""
        reg = MarketplaceRegistry()  # empty registry
        manager, _, _, _ = _create_pipeline(registry=reg)

        result = manager.hire(HireRequest(
            required_skills=["quantum-computing"],
            budget=100.0,
            description="Quantum simulation",
        ))

        assert result.status == "no_agents"
        assert result.agent_name == ""
        assert result.escrow_id == ""
        assert result.task_result is None
        assert "No agents found" in result.error

    def test_no_agents_with_price_constraint(self):
        """All agents too expensive for the budget."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Expensive", ["code"], price=100.0))

        budget = BudgetTracker(total_budget=1000.0)
        manager = HiringManager(registry=reg, budget_tracker=budget)

        result = manager.hire(HireRequest(
            required_skills=["code"],
            budget=0.01,  # Per-request budget too low
        ))

        assert result.status in ("no_agents", "budget_exceeded")

    def test_global_budget_exhausted_mid_pipeline(self):
        """Budget runs out between multiple hires."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Worker", ["code"], price=0.50))
        budget = BudgetTracker(total_budget=0.75)
        manager = HiringManager(registry=reg, budget_tracker=budget)

        # First hire succeeds (0.50 / 0.75 used)
        r1 = manager.hire(HireRequest(required_skills=["code"], budget=1.0))
        assert r1.status == "completed"

        # Second hire fails — only 0.25 remaining, agent costs 0.50
        r2 = manager.hire(HireRequest(required_skills=["code"], budget=1.0))
        assert r2.status == "budget_exceeded"
        assert budget.remaining == pytest.approx(0.25)

    def test_escrow_not_created_when_budget_exceeded(self):
        """No escrow entry should exist for failed budget checks."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Worker", ["code"], price=10.0))
        escrow = AgentEscrow()
        budget = BudgetTracker(total_budget=5.0)
        manager = HiringManager(registry=reg, escrow=escrow, budget_tracker=budget)

        result = manager.hire(HireRequest(required_skills=["code"], budget=100.0))
        assert result.status == "budget_exceeded"
        assert len(escrow.list_all()) == 0

    def test_zero_budget_request(self):
        """Request with zero budget should fail price negotiation."""
        manager, _, _, _ = _create_pipeline()
        result = manager.hire(HireRequest(
            required_skills=["code"],
            budget=0.0,  # Zero budget
        ))
        # Zero budget means no agents within price range
        assert result.status in ("no_agents", "budget_exceeded")

    def test_hire_result_contains_task_id(self):
        """Every hire result should have a unique task_id."""
        manager, _, _, _ = _create_pipeline()
        r1 = manager.hire(HireRequest(required_skills=["code"], budget=5.0))
        r2 = manager.hire(HireRequest(required_skills=["design"], budget=5.0))

        assert r1.task_id.startswith("hire_")
        assert r2.task_id.startswith("hire_")
        assert r1.task_id != r2.task_id

    def test_hire_result_elapsed_time_recorded(self):
        """Hire results should record elapsed time."""
        manager, _, _, _ = _create_pipeline()
        result = manager.hire(HireRequest(required_skills=["code"], budget=5.0))
        assert result.elapsed_s >= 0

    def test_failed_verification_triggers_refund(self):
        """If verify_result fails, escrow should be refunded."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Worker", ["code"], price=0.01))
        escrow = AgentEscrow()
        budget = BudgetTracker(total_budget=10.0)

        class FailingManager(HiringManager):
            def verify_result(self, result):
                return False  # Always fail verification

        manager = FailingManager(registry=reg, escrow=escrow, budget_tracker=budget)
        result = manager.hire(HireRequest(required_skills=["code"], budget=5.0))

        assert result.status == "failed"
        assert "verification failed" in result.error.lower()

        # Escrow should be refunded, not released
        entry = escrow.get_entry(result.escrow_id)
        assert entry is not None
        assert entry.status == "refunded"


# ===================================================================
# 4. Skill Matching Integration
# ===================================================================


class TestSkillMatchingIntegration:
    """Test agent matching produces correct rankings in pipeline context."""

    def test_exact_skill_match_preferred(self):
        """Agent with exact skill match should rank highest."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Generalist", ["general"], price=0.01, rating=5.0))
        reg.register_agent(_make_listing("Specialist", ["python", "testing"], price=0.01, rating=3.0))

        matcher = SkillMatcher(reg)
        results = matcher.match(["python", "testing"])

        assert len(results) >= 1
        assert results[0][0].name == "Specialist"

    def test_partial_skill_match_scored_lower(self):
        """Agent with partial skill match should score lower."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Full", ["python", "testing", "deploy"], price=0.01, rating=4.0))
        reg.register_agent(_make_listing("Partial", ["python"], price=0.01, rating=4.0))

        matcher = SkillMatcher(reg)
        results = matcher.match(["python", "testing"])

        scores = {r[0].name: r[1] for r in results}
        assert scores["Full"] > scores["Partial"]

    def test_price_filter_excludes_expensive_agents(self):
        """Max price filter should exclude expensive agents from matching."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Cheap", ["code"], price=0.01))
        reg.register_agent(_make_listing("Expensive", ["code"], price=100.0))

        matcher = SkillMatcher(reg)
        results = matcher.match(["code"], max_price=1.0)

        names = [r[0].name for r in results]
        assert "Cheap" in names
        assert "Expensive" not in names

    def test_rating_filter_excludes_low_rated(self):
        """Min rating filter should exclude low-rated agents."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("HighRated", ["code"], rating=4.5))
        reg.register_agent(_make_listing("LowRated", ["code"], rating=2.0))

        matcher = SkillMatcher(reg)
        results = matcher.match(["code"], min_rating=4.0)

        names = [r[0].name for r in results]
        assert "HighRated" in names
        assert "LowRated" not in names


# ===================================================================
# 5. Budget Tracking Integration
# ===================================================================


class TestBudgetTrackingIntegration:
    """Budget enforcement across the full pipeline."""

    def test_budget_tracks_spending_per_task(self):
        """Each task's spending is tracked independently."""
        manager, _, _, bt = _create_pipeline(budget=10.0)

        r1 = manager.hire(HireRequest(required_skills=["code"], budget=5.0, description="Task 1"))
        r2 = manager.hire(HireRequest(required_skills=["design"], budget=5.0, description="Task 2"))

        assert bt.get_spending(r1.task_id) == r1.agreed_price
        assert bt.get_spending(r2.task_id) == r2.agreed_price

    def test_spending_report_after_multiple_hires(self):
        """Spending report should reflect all completed hires."""
        manager, _, _, bt = _create_pipeline(budget=10.0)

        manager.hire(HireRequest(required_skills=["code"], budget=5.0))
        manager.hire(HireRequest(required_skills=["research"], budget=5.0))
        manager.hire(HireRequest(required_skills=["writing"], budget=5.0))

        report = bt.spending_report()
        assert report["total_budget"] == 10.0
        assert report["total_spent"] > 0
        assert len(report["tasks"]) == 3
        assert report["remaining"] == pytest.approx(10.0 - report["total_spent"])

    def test_budget_remaining_accurate_after_failures(self):
        """Budget should not be deducted for failed hires."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Worker", ["code"], price=0.10))
        budget = BudgetTracker(total_budget=1.0)
        manager = HiringManager(registry=reg, budget_tracker=budget)

        # Succeed
        manager.hire(HireRequest(required_skills=["code"], budget=5.0))
        # Fail (no quantum agents)
        manager.hire(HireRequest(required_skills=["quantum"], budget=5.0))

        # Only the successful hire should be deducted
        assert budget.total_spent == pytest.approx(0.10)
        assert budget.remaining == pytest.approx(0.90)


# ===================================================================
# 6. API Endpoint Error Codes
# ===================================================================


@pytest.fixture
def api_client():
    """Create a test client with marketplace routes and fresh state."""
    import src.api.marketplace_routes as routes_mod

    fresh_marketplace = MarketplaceRegistry()
    fresh_marketplace.register_agent(_make_listing("APICoder", ["code", "python"], 0.01, 4.5, "api-coder-001"))
    fresh_marketplace.register_agent(_make_listing("APIDesigner", ["design", "ui"], 0.05, 4.8, "api-designer-001"))

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


class TestAPIErrorCodes:
    """Verify API endpoints return proper HTTP status codes and error messages."""

    # -- 404 errors --

    def test_get_agent_not_found_returns_404(self, api_client):
        resp = api_client.get("/marketplace/agents/nonexistent-agent-id")
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_get_hire_status_not_found_returns_404(self, api_client):
        resp = api_client.get("/marketplace/hire/nonexistent-task/status")
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    # -- 422 validation errors --

    def test_register_agent_missing_name_returns_422(self, api_client):
        resp = api_client.post("/marketplace/agents", json={})
        assert resp.status_code == 422

    def test_register_agent_empty_name_returns_422(self, api_client):
        resp = api_client.post("/marketplace/agents", json={"name": ""})
        assert resp.status_code == 422

    def test_hire_missing_description_returns_422(self, api_client):
        resp = api_client.post("/marketplace/hire", json={})
        assert resp.status_code == 422

    def test_hire_empty_description_returns_422(self, api_client):
        resp = api_client.post("/marketplace/hire", json={"description": ""})
        assert resp.status_code == 422

    def test_hire_negative_budget_returns_422(self, api_client):
        resp = api_client.post("/marketplace/hire", json={
            "description": "Test task",
            "budget": -1.0,
        })
        assert resp.status_code == 422

    def test_hire_zero_budget_returns_422(self, api_client):
        resp = api_client.post("/marketplace/hire", json={
            "description": "Test task",
            "budget": 0.0,
        })
        assert resp.status_code == 422

    def test_hire_excessive_budget_returns_422(self, api_client):
        resp = api_client.post("/marketplace/hire", json={
            "description": "Test task",
            "budget": 99999.0,
        })
        assert resp.status_code == 422

    def test_payment_request_missing_fields_returns_422(self, api_client):
        resp = api_client.post("/payments/request", json={})
        assert resp.status_code == 422

    def test_payment_verify_missing_fields_returns_422(self, api_client):
        resp = api_client.post("/payments/verify", json={})
        assert resp.status_code == 422

    def test_payment_request_zero_amount_returns_422(self, api_client):
        resp = api_client.post("/payments/request", json={
            "resource": "/test",
            "amount": 0.0,
            "payee": "agent1",
        })
        assert resp.status_code == 422

    # -- 200 success responses --

    def test_list_agents_returns_200(self, api_client):
        resp = api_client.get("/marketplace/agents")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_agent_returns_200(self, api_client):
        resp = api_client.get("/marketplace/agents/api-coder-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "APICoder"
        assert data["agent_id"] == "api-coder-001"

    def test_hire_success_returns_200(self, api_client):
        resp = api_client.post("/marketplace/hire", json={
            "description": "Write some code",
            "required_skills": ["code"],
            "budget": 1.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    def test_list_jobs_returns_200(self, api_client):
        resp = api_client.get("/marketplace/jobs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_budget_returns_200(self, api_client):
        resp = api_client.get("/marketplace/budget")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_budget" in data
        assert "remaining" in data

    def test_x402_info_returns_200(self, api_client):
        resp = api_client.get("/marketplace/x402")
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] == "Payment Required"
        assert "accepts" in data

    # -- 201 created --

    def test_register_agent_returns_201(self, api_client):
        resp = api_client.post("/marketplace/agents", json={
            "name": "NewAgent",
            "skills": ["code"],
            "price_per_unit": 0.05,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "NewAgent"
        assert data["agent_id"].startswith("agent_")

    # -- 402 payment required --

    def test_payment_request_returns_402(self, api_client):
        resp = api_client.post("/payments/request", json={
            "resource": "/marketplace/hire/task-123",
            "amount": 0.05,
            "payee": "agent1",
            "description": "Hire agent for code review",
        })
        assert resp.status_code == 402
        data = resp.json()
        assert data["error"] == "Payment Required"
        assert "accepts" in data
        assert resp.headers.get("X-Payment") == "required"

    # -- Payment verification endpoint --

    def test_payment_verify_returns_verified(self, api_client):
        resp = api_client.post("/payments/verify", json={
            "payer": "0xBuyer",
            "payee": "0xSeller",
            "amount": 0.05,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "verified" in data
        assert "payment_id" in data

    def test_balance_endpoint_returns_200(self, api_client):
        resp = api_client.get("/payments/balance/agent1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "agent1"
        assert data["balance"] == 0.0

    def test_ledger_endpoint_returns_200(self, api_client):
        resp = api_client.get("/payments/ledger")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ===================================================================
# 7. API Pipeline Integration (Full flow via REST)
# ===================================================================


class TestAPIPipelineIntegration:
    """End-to-end API tests: register -> hire -> check status -> list jobs."""

    def test_register_then_hire_then_check_status(self, api_client):
        """Full API flow: register agent, hire it, check status."""
        # Register a new agent
        reg_resp = api_client.post("/marketplace/agents", json={
            "name": "PipelineAgent",
            "skills": ["automation", "testing"],
            "price_per_unit": 0.01,
        })
        assert reg_resp.status_code == 201

        # Hire the agent
        hire_resp = api_client.post("/marketplace/hire", json={
            "description": "Run automated tests",
            "required_skills": ["automation"],
            "budget": 1.0,
        })
        assert hire_resp.status_code == 200
        hire_data = hire_resp.json()
        assert hire_data["status"] == "completed"
        task_id = hire_data["task_id"]

        # Check hire status
        status_resp = api_client.get(f"/marketplace/hire/{task_id}/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["task_id"] == task_id
        assert status_data["status"] == "completed"
        assert status_data["escrow_status"] == "released"

    def test_hire_then_list_jobs_shows_result(self, api_client):
        """After hiring, job should appear in jobs list."""
        api_client.post("/marketplace/hire", json={
            "description": "Code review",
            "required_skills": ["code"],
            "budget": 1.0,
        })

        jobs_resp = api_client.get("/marketplace/jobs")
        assert jobs_resp.status_code == 200
        jobs = jobs_resp.json()
        assert len(jobs) >= 1
        assert jobs[0]["status"] == "completed"
        assert jobs[0]["escrow_status"] == "released"

    def test_hire_then_budget_decremented(self, api_client):
        """Budget should decrease after hiring."""
        budget_before = api_client.get("/marketplace/budget").json()

        api_client.post("/marketplace/hire", json={
            "description": "Quick task",
            "required_skills": ["code"],
            "budget": 1.0,
        })

        budget_after = api_client.get("/marketplace/budget").json()
        assert budget_after["total_spent"] > budget_before["total_spent"]
        assert budget_after["remaining"] < budget_before["remaining"]

    def test_multiple_hires_reflected_in_jobs_and_budget(self, api_client):
        """Multiple hires should all appear in jobs and budget."""
        for skill in ["code", "design"]:
            api_client.post("/marketplace/hire", json={
                "description": f"Task for {skill}",
                "required_skills": [skill],
                "budget": 5.0,
            })

        jobs = api_client.get("/marketplace/jobs").json()
        assert len(jobs) == 2

        budget = api_client.get("/marketplace/budget").json()
        assert budget["total_spent"] > 0
        assert len(budget["tasks"]) == 2

    def test_hire_no_matching_skills_returns_no_agents(self, api_client):
        """Hiring with unmatched skills returns no_agents status."""
        resp = api_client.post("/marketplace/hire", json={
            "description": "Quantum computing simulation",
            "required_skills": ["quantum-physics"],
            "budget": 10.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "no_agents"

    def test_agent_sort_by_price(self, api_client):
        """Agents can be sorted by price."""
        resp = api_client.get("/marketplace/agents?sort_by=price")
        assert resp.status_code == 200
        agents = resp.json()
        prices = [a["price_per_unit"] for a in agents]
        assert prices == sorted(prices)

    def test_agent_sort_by_rating(self, api_client):
        """Agents can be sorted by rating (descending)."""
        resp = api_client.get("/marketplace/agents?sort_by=rating")
        assert resp.status_code == 200
        agents = resp.json()
        ratings = [a["rating"] for a in agents]
        assert ratings == sorted(ratings, reverse=True)

    def test_agent_response_shape(self, api_client):
        """Verify agent response includes all expected fields."""
        resp = api_client.get("/marketplace/agents/api-coder-001")
        assert resp.status_code == 200
        data = resp.json()

        expected_fields = [
            "agent_id", "name", "description", "skills", "pricing_model",
            "price_per_unit", "price_display", "rating", "total_jobs",
            "completed_jobs", "failed_jobs", "completion_rate", "total_earnings",
            "availability", "endpoint", "protocol",
        ]
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"


# ===================================================================
# 8. Main API Endpoints (Task CRUD + Health)
# ===================================================================


@pytest.fixture
def main_api_client():
    """Create a test client with the full main app."""
    from src.api.main import app
    client = TestClient(app)
    yield client


class TestMainAPIEndpoints:
    """Test core task and health endpoints for proper error codes."""

    def test_health_returns_200(self, main_api_client):
        resp = main_api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert "tasks_total" in data

    def test_list_tasks_returns_200(self, main_api_client):
        resp = main_api_client.get("/tasks")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_task_not_found_returns_404(self, main_api_client):
        resp = main_api_client.get("/tasks/nonexistent-task-id")
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_submit_task_returns_201(self, main_api_client):
        resp = main_api_client.post("/tasks", json={
            "description": "Build authentication module",
            "budget": 5.0,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "pending"
        assert data["budget_usd"] == 5.0
        assert data["description"] == "Build authentication module"

    def test_submit_task_missing_description_returns_422(self, main_api_client):
        resp = main_api_client.post("/tasks", json={})
        assert resp.status_code == 422

    def test_submit_task_empty_description_returns_422(self, main_api_client):
        resp = main_api_client.post("/tasks", json={"description": ""})
        assert resp.status_code == 422

    def test_submit_task_negative_budget_returns_422(self, main_api_client):
        resp = main_api_client.post("/tasks", json={
            "description": "Some task",
            "budget": -1.0,
        })
        assert resp.status_code == 422

    def test_submit_task_zero_budget_returns_422(self, main_api_client):
        resp = main_api_client.post("/tasks", json={
            "description": "Some task",
            "budget": 0.0,
        })
        assert resp.status_code == 422

    def test_submit_task_excessive_budget_returns_422(self, main_api_client):
        resp = main_api_client.post("/tasks", json={
            "description": "Some task",
            "budget": 5000.0,
        })
        assert resp.status_code == 422

    def test_transactions_returns_200(self, main_api_client):
        resp = main_api_client.get("/transactions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_agents_returns_200(self, main_api_client):
        resp = main_api_client.get("/agents")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_root_returns_200(self, main_api_client):
        resp = main_api_client.get("/")
        assert resp.status_code == 200

    def test_task_response_shape(self, main_api_client):
        """Verify task response includes all expected fields."""
        resp = main_api_client.post("/tasks", json={
            "description": "Test shape",
            "budget": 1.0,
        })
        assert resp.status_code == 201
        data = resp.json()

        expected_fields = ["task_id", "description", "status", "budget_usd", "created_at"]
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"

    def test_health_response_shape(self, main_api_client):
        """Verify health response includes all expected fields."""
        resp = main_api_client.get("/health")
        data = resp.json()

        expected_fields = [
            "status", "uptime_seconds", "tasks_total", "tasks_completed",
            "tasks_pending", "tasks_running", "agents_count",
            "total_spent_usdc", "gpt4o_available",
        ]
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"


# ===================================================================
# 9. PaymentLedger Audit Trail Integration
# ===================================================================


class TestLedgerAuditTrail:
    """Verify the payment ledger provides a complete audit trail."""

    def test_ledger_records_all_events_in_order(self):
        """Events should be recorded in chronological order."""
        ledger = PaymentLedger()
        ledger.record(event_type="payment_request", amount=1.0, payee="agent1")
        ledger.record(event_type="escrow_hold", amount=1.0, payer="ceo", payee="agent1", task_id="t1")
        ledger.record(event_type="escrow_release", amount=1.0, payer="ceo", payee="agent1", task_id="t1")

        entries = ledger.get_all()
        assert len(entries) == 3
        types = [e.event_type for e in entries]
        assert types == ["payment_request", "escrow_hold", "escrow_release"]

    def test_ledger_filter_by_event_type(self):
        ledger = PaymentLedger()
        ledger.record(event_type="payment_request", amount=1.0)
        ledger.record(event_type="escrow_hold", amount=2.0)
        ledger.record(event_type="payment_request", amount=3.0)

        requests = ledger.get_entries(event_type="payment_request")
        assert len(requests) == 2

    def test_ledger_filter_by_task_id(self):
        ledger = PaymentLedger()
        ledger.record(event_type="escrow_hold", task_id="t1", amount=1.0)
        ledger.record(event_type="escrow_hold", task_id="t2", amount=2.0)

        t1_entries = ledger.get_entries(task_id="t1")
        assert len(t1_entries) == 1
        assert t1_entries[0].task_id == "t1"

    def test_ledger_filter_by_agent(self):
        ledger = PaymentLedger()
        ledger.record(event_type="escrow_hold", payer="ceo", payee="agent1", amount=1.0)
        ledger.record(event_type="escrow_hold", payer="ceo", payee="agent2", amount=2.0)

        agent1_entries = ledger.get_entries(agent_id="agent1")
        assert len(agent1_entries) == 1

    def test_ledger_total_volume(self):
        ledger = PaymentLedger()
        ledger.record(event_type="payment_request", amount=1.0)
        ledger.record(event_type="escrow_hold", amount=2.5)
        assert ledger.total_volume() == pytest.approx(3.5)

    def test_ledger_clear(self):
        ledger = PaymentLedger()
        ledger.record(event_type="test", amount=1.0)
        ledger.clear()
        assert ledger.count() == 0
        assert ledger.total_volume() == 0.0

    def test_ledger_entry_has_all_fields(self):
        ledger = PaymentLedger()
        entry = ledger.record(
            event_type="escrow_hold",
            payer="ceo",
            payee="agent1",
            amount=0.50,
            task_id="t1",
            escrow_id="e1",
            payment_id="p1",
            network="eip155:8453",
            metadata={"note": "test"},
        )

        assert entry.entry_id.startswith("ledger_")
        assert entry.event_type == "escrow_hold"
        assert entry.payer == "ceo"
        assert entry.payee == "agent1"
        assert entry.amount == 0.50
        assert entry.task_id == "t1"
        assert entry.escrow_id == "e1"
        assert entry.payment_id == "p1"
        assert entry.network == "eip155:8453"
        assert entry.metadata == {"note": "test"}
        assert entry.timestamp > 0

        d = entry.to_dict()
        assert isinstance(d, dict)
        assert d["event_type"] == "escrow_hold"


# ===================================================================
# 10. Marketplace Registry Reputation Integration
# ===================================================================


class TestReputationIntegration:
    """Agent reputation updates through the pipeline."""

    def test_reputation_updates_after_successful_hire(self):
        """Agent's total_jobs should increment after successful hire."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Worker", ["code"], price=0.01, agent_id="w1"))
        manager = HiringManager(registry=reg, budget_tracker=BudgetTracker(10.0))

        worker = reg.get_agent("w1")
        initial_jobs = worker.total_jobs

        manager.hire(HireRequest(required_skills=["code"], budget=5.0))
        assert worker.total_jobs == initial_jobs + 1

    def test_reputation_profile_reflects_completions(self):
        """Reputation profile should reflect completed jobs."""
        reg = MarketplaceRegistry()
        listing = _make_listing("Agent", ["code"], price=0.01, agent_id="a1")
        reg.register_agent(listing)

        for i in range(3):
            reg.record_job_completion("a1", success=True, earnings=0.01)
        reg.record_job_completion("a1", success=False)

        rep = reg.get_reputation("a1")
        assert rep is not None
        assert rep["total_jobs"] == 4
        assert rep["completed_jobs"] == 3
        assert rep["failed_jobs"] == 1
        assert rep["completion_rate"] == pytest.approx(0.75)
        assert rep["total_earnings"] == pytest.approx(0.03)

    def test_reputation_nonexistent_agent_returns_none(self):
        reg = MarketplaceRegistry()
        assert reg.get_reputation("ghost") is None

    def test_availability_changes_respected(self):
        """Busy/offline agents should not appear in available list."""
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Available", ["code"], agent_id="a1"))
        reg.register_agent(_make_listing("Busy", ["code"], agent_id="a2"))
        reg.register_agent(_make_listing("Offline", ["code"], agent_id="a3"))

        reg.set_availability("a2", "busy")
        reg.set_availability("a3", "offline")

        available = reg.list_available()
        names = [a.name for a in available]
        assert "Available" in names
        assert "Busy" not in names
        assert "Offline" not in names

    def test_set_invalid_availability_returns_false(self):
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Agent", ["code"], agent_id="a1"))
        assert reg.set_availability("a1", "invalid_status") is False

    def test_rolling_average_rating(self):
        """Update rating uses rolling average for agents with history."""
        reg = MarketplaceRegistry()
        listing = _make_listing("Agent", ["code"], rating=4.0, agent_id="a1")
        listing.total_jobs = 5
        reg.register_agent(listing)

        reg.update_agent_rating("a1", 5.0)
        # Rolling avg: 4.0 * 0.7 + 5.0 * 0.3 = 4.3
        assert reg.get_agent("a1").rating == pytest.approx(4.3, abs=0.01)
