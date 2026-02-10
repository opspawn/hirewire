"""Tests for Agent Marketplace, x402 payment layer, hiring flow, and API.

Covers:
- AgentListing CRUD and discovery
- SkillMatcher ranking and filtering
- x402 PaymentConfig, 402 response format compliance
- X402PaymentGate verification flow
- AgentEscrow lifecycle (hold → release/refund)
- HiringManager full lifecycle
- BudgetTracker enforcement
- Marketplace REST API endpoints
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.marketplace import (
    AgentListing,
    MarketplaceRegistry,
    SkillMatcher,
    marketplace,
)
from src.marketplace.x402 import (
    PaymentConfig,
    PaymentProof,
    X402PaymentGate,
    EscrowEntry,
    AgentEscrow,
)
from src.marketplace.hiring import (
    HireRequest,
    HireResult,
    BudgetTracker,
    HiringManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_listing(
    name: str = "TestAgent",
    skills: list[str] | None = None,
    price: float = 0.01,
    rating: float = 4.0,
    agent_id: str | None = None,
) -> AgentListing:
    listing = AgentListing(
        name=name,
        description=f"A test agent: {name}",
        skills=skills or ["code", "testing"],
        pricing_model="per-task",
        price_per_unit=price,
        rating=rating,
        endpoint=f"http://localhost:9000/{name.lower()}",
    )
    if agent_id:
        listing.agent_id = agent_id
    return listing


def _fresh_registry() -> MarketplaceRegistry:
    return MarketplaceRegistry()


def _seeded_registry() -> MarketplaceRegistry:
    """A registry with 5 pre-seeded agents."""
    reg = MarketplaceRegistry()
    reg.register_agent(_make_listing("Coder", ["code", "python", "testing"], 0.02, 4.5))
    reg.register_agent(_make_listing("Designer", ["design", "ui", "ux"], 0.05, 4.8))
    reg.register_agent(_make_listing("Researcher", ["research", "analysis", "web-search"], 0.01, 4.2))
    reg.register_agent(_make_listing("DevOps", ["deployment", "docker", "ci-cd"], 0.03, 3.9))
    reg.register_agent(_make_listing("Writer", ["writing", "docs", "content"], 0.00, 4.0))
    return reg


# ===================================================================
# 1. AgentListing — creation and properties
# ===================================================================


class TestAgentListing:
    def test_create_default(self):
        listing = AgentListing()
        assert listing.agent_id.startswith("agent_")
        assert listing.name == ""
        assert listing.rating == 0.0
        assert listing.total_jobs == 0

    def test_create_with_values(self):
        listing = _make_listing("Alpha", ["code", "deploy"], 0.05, 4.7)
        assert listing.name == "Alpha"
        assert listing.price_per_unit == 0.05
        assert listing.rating == 4.7

    def test_price_display_per_task(self):
        listing = _make_listing(price=0.05)
        assert "$0.0500/task" in listing.price_display

    def test_price_display_per_token(self):
        listing = AgentListing(pricing_model="per-token", price_per_unit=0.001)
        assert "1K tokens" in listing.price_display

    def test_matches_skill_positive(self):
        listing = _make_listing(skills=["python", "testing"])
        assert listing.matches_skill("python") is True
        assert listing.matches_skill("Python") is True  # case insensitive

    def test_matches_skill_negative(self):
        listing = _make_listing(skills=["python", "testing"])
        assert listing.matches_skill("design") is False

    def test_matches_skill_partial(self):
        listing = _make_listing(skills=["web-search"])
        assert listing.matches_skill("search") is True

    def test_to_dict(self):
        listing = _make_listing("Beta")
        d = listing.to_dict()
        assert d["name"] == "Beta"
        assert "agent_id" in d
        assert "skills" in d
        assert isinstance(d["skills"], list)

    def test_registered_at_auto(self):
        listing = AgentListing()
        assert listing.registered_at > 0


# ===================================================================
# 2. MarketplaceRegistry — CRUD
# ===================================================================


class TestMarketplaceRegistry:
    def test_register_agent(self):
        reg = _fresh_registry()
        listing = _make_listing("Alpha")
        result = reg.register_agent(listing)
        assert result is listing
        assert reg.count() == 1

    def test_get_agent(self):
        reg = _fresh_registry()
        listing = _make_listing("Beta", agent_id="beta-001")
        reg.register_agent(listing)
        found = reg.get_agent("beta-001")
        assert found is listing

    def test_get_agent_not_found(self):
        reg = _fresh_registry()
        assert reg.get_agent("nonexistent") is None

    def test_get_agent_by_name(self):
        reg = _fresh_registry()
        listing = _make_listing("Gamma")
        reg.register_agent(listing)
        found = reg.get_agent_by_name("Gamma")
        assert found is listing

    def test_get_agent_by_name_not_found(self):
        reg = _fresh_registry()
        assert reg.get_agent_by_name("Nobody") is None

    def test_unregister_agent(self):
        reg = _fresh_registry()
        listing = _make_listing("Delta", agent_id="delta-001")
        reg.register_agent(listing)
        assert reg.unregister_agent("delta-001") is True
        assert reg.count() == 0

    def test_unregister_agent_not_found(self):
        reg = _fresh_registry()
        assert reg.unregister_agent("nonexistent") is False

    def test_list_all(self):
        reg = _seeded_registry()
        assert len(reg.list_all()) == 5

    def test_count(self):
        reg = _seeded_registry()
        assert reg.count() == 5

    def test_clear(self):
        reg = _seeded_registry()
        reg.clear()
        assert reg.count() == 0

    def test_update_rating(self):
        reg = _fresh_registry()
        listing = _make_listing("RateMe", agent_id="rate-001")
        reg.register_agent(listing)
        assert reg.update_rating("rate-001", 4.9) is True
        assert reg.get_agent("rate-001").rating == 4.9

    def test_update_rating_clamped(self):
        reg = _fresh_registry()
        listing = _make_listing("ClampMe", agent_id="clamp-001")
        reg.register_agent(listing)
        reg.update_rating("clamp-001", 10.0)
        assert reg.get_agent("clamp-001").rating == 5.0
        reg.update_rating("clamp-001", -5.0)
        assert reg.get_agent("clamp-001").rating == 0.0

    def test_update_rating_not_found(self):
        reg = _fresh_registry()
        assert reg.update_rating("nope", 3.0) is False

    def test_increment_jobs(self):
        reg = _fresh_registry()
        listing = _make_listing("Worker", agent_id="worker-001")
        reg.register_agent(listing)
        assert reg.increment_jobs("worker-001") is True
        assert reg.get_agent("worker-001").total_jobs == 1

    def test_increment_jobs_not_found(self):
        reg = _fresh_registry()
        assert reg.increment_jobs("nope") is False


# ===================================================================
# 3. MarketplaceRegistry — discovery
# ===================================================================


class TestDiscovery:
    def test_discover_by_skill(self):
        reg = _seeded_registry()
        results = reg.discover_agents("code")
        assert len(results) == 1
        assert results[0].name == "Coder"

    def test_discover_by_description(self):
        reg = _seeded_registry()
        results = reg.discover_agents("test agent")
        assert len(results) == 5  # all match "A test agent: ..."

    def test_discover_by_name(self):
        reg = _seeded_registry()
        results = reg.discover_agents("Designer")
        assert len(results) == 1

    def test_discover_with_max_price(self):
        reg = _seeded_registry()
        results = reg.discover_agents("test agent", max_price=0.02)
        names = [r.name for r in results]
        assert "Designer" not in names  # $0.05 too expensive
        assert "Writer" in names  # $0.00

    def test_discover_no_results(self):
        reg = _seeded_registry()
        results = reg.discover_agents("quantum-physics")
        assert len(results) == 0


# ===================================================================
# 4. SkillMatcher
# ===================================================================


class TestSkillMatcher:
    def test_match_single_skill(self):
        reg = _seeded_registry()
        matcher = SkillMatcher(reg)
        results = matcher.match(["code"])
        assert len(results) >= 1
        assert results[0][0].name == "Coder"

    def test_match_multiple_skills(self):
        reg = _seeded_registry()
        matcher = SkillMatcher(reg)
        results = matcher.match(["code", "testing"])
        assert len(results) >= 1
        # Coder has both skills, should rank highest
        assert results[0][0].name == "Coder"

    def test_match_with_rating_filter(self):
        reg = _seeded_registry()
        matcher = SkillMatcher(reg)
        results = matcher.match(["deployment"], min_rating=4.0)
        # DevOps has rating 3.9, should be excluded
        names = [r[0].name for r in results]
        assert "DevOps" not in names

    def test_match_with_max_price(self):
        reg = _seeded_registry()
        matcher = SkillMatcher(reg)
        results = matcher.match(["design"], max_price=0.03)
        names = [r[0].name for r in results]
        assert "Designer" not in names  # $0.05 too expensive

    def test_match_top_n(self):
        reg = _seeded_registry()
        matcher = SkillMatcher(reg)
        results = matcher.match(["test agent"], top_n=2)
        assert len(results) <= 2

    def test_match_no_skills_returns_all(self):
        reg = _seeded_registry()
        matcher = SkillMatcher(reg)
        results = matcher.match([])
        assert len(results) == 5

    def test_match_score_range(self):
        reg = _seeded_registry()
        matcher = SkillMatcher(reg)
        results = matcher.match(["code", "testing"])
        for _, score in results:
            assert 0.0 < score <= 2.0

    def test_best_match(self):
        reg = _seeded_registry()
        matcher = SkillMatcher(reg)
        result = matcher.best_match(["code", "python"])
        assert result is not None
        assert result[0].name == "Coder"

    def test_best_match_no_results(self):
        reg = _fresh_registry()
        matcher = SkillMatcher(reg)
        assert matcher.best_match(["quantum"]) is None


# ===================================================================
# 5. PaymentConfig
# ===================================================================


class TestPaymentConfig:
    def test_default_config(self):
        cfg = PaymentConfig()
        assert cfg.network == "eip155:8453"
        assert cfg.price == 0.0
        assert cfg.asset == "USDC"

    def test_to_accepts_entry(self):
        cfg = PaymentConfig(
            price=0.01,
            pay_to="0xABCDEF",
            facilitator_url="https://fac.example.com",
        )
        entry = cfg.to_accepts_entry("/api/test", "Test resource")
        assert entry["scheme"] == "exact"
        assert entry["network"] == "eip155:8453"
        assert entry["maxAmountRequired"] == "10000"  # 0.01 * 1_000_000
        assert entry["payTo"] == "0xABCDEF"
        assert entry["resource"] == "/api/test"
        assert entry["description"] == "Test resource"
        assert entry["mimeType"] == "application/json"
        assert entry["extra"]["facilitatorUrl"] == "https://fac.example.com"
        assert entry["extra"]["name"] == "USDC"
        assert entry["requiredDeadlineSeconds"] == 300

    def test_zero_price(self):
        cfg = PaymentConfig(price=0.0)
        entry = cfg.to_accepts_entry()
        assert entry["maxAmountRequired"] == "0"

    def test_custom_deadline(self):
        cfg = PaymentConfig(deadline_seconds=600)
        entry = cfg.to_accepts_entry()
        assert entry["requiredDeadlineSeconds"] == 600


# ===================================================================
# 6. X402PaymentGate
# ===================================================================


class TestX402PaymentGate:
    def test_create_402_response_format(self):
        gate = X402PaymentGate(PaymentConfig(price=0.05, pay_to="0x123"))
        resp = gate.create_402_response("/test", "Pay up")
        assert resp["error"] == "Payment Required"
        assert isinstance(resp["accepts"], list)
        assert len(resp["accepts"]) == 1
        accept = resp["accepts"][0]
        assert accept["scheme"] == "exact"
        assert accept["payTo"] == "0x123"

    def test_create_402_with_price_override(self):
        gate = X402PaymentGate(PaymentConfig(price=0.01))
        resp = gate.create_402_response(price_override=0.10)
        amount = resp["accepts"][0]["maxAmountRequired"]
        assert amount == "100000"  # 0.10 USDC

    def test_verify_payment_success(self):
        gate = X402PaymentGate(PaymentConfig(price=0.01, pay_to="0xPayee"))
        proof = PaymentProof(payer="0xPayer", payee="0xPayee", amount=0.01)
        assert gate.verify_payment(proof) is True
        assert proof.verified is True

    def test_verify_payment_wrong_payee(self):
        gate = X402PaymentGate(PaymentConfig(price=0.01, pay_to="0xPayee"))
        proof = PaymentProof(payer="0xPayer", payee="0xWrong", amount=0.01)
        assert gate.verify_payment(proof) is False

    def test_verify_payment_insufficient_amount(self):
        gate = X402PaymentGate(PaymentConfig(price=0.10, pay_to="0xPayee"))
        proof = PaymentProof(payer="0xPayer", payee="0xPayee", amount=0.05)
        assert gate.verify_payment(proof) is False

    def test_verify_payment_no_payee_required(self):
        gate = X402PaymentGate(PaymentConfig(price=0.01, pay_to=""))
        proof = PaymentProof(payer="0xPayer", payee="0xAnyone", amount=0.01)
        assert gate.verify_payment(proof) is True

    def test_payment_history(self):
        gate = X402PaymentGate(PaymentConfig(price=0.0, pay_to=""))
        p1 = PaymentProof(payer="A", payee="B", amount=0.01)
        p2 = PaymentProof(payer="C", payee="D", amount=0.02)
        gate.verify_payment(p1)
        gate.verify_payment(p2)
        assert len(gate.payment_history()) == 2
        assert len(gate.payment_history(payer="A")) == 1

    def test_is_paid(self):
        gate = X402PaymentGate(PaymentConfig(price=0.0))
        assert gate.is_paid("/test") is False
        proof = PaymentProof(payer="A", payee="B", amount=0.01, verified=True)
        gate.record_verified_payment("/test", proof)
        assert gate.is_paid("/test") is True

    def test_total_collected(self):
        gate = X402PaymentGate(PaymentConfig(price=0.0, pay_to=""))
        gate.verify_payment(PaymentProof(payer="A", payee="", amount=0.05))
        gate.verify_payment(PaymentProof(payer="B", payee="", amount=0.10))
        assert gate.total_collected() == pytest.approx(0.15)


# ===================================================================
# 7. AgentEscrow
# ===================================================================


class TestAgentEscrow:
    def test_hold_payment(self):
        escrow = AgentEscrow()
        entry = escrow.hold_payment("ceo", "agent1", 0.05, "task-1")
        assert entry.status == "held"
        assert entry.payer == "ceo"
        assert entry.payee == "agent1"
        assert entry.amount == 0.05

    def test_release_on_completion(self):
        escrow = AgentEscrow()
        entry = escrow.hold_payment("ceo", "agent1", 0.05, "task-1")
        released = escrow.release_on_completion(entry.escrow_id)
        assert released is not None
        assert released.status == "released"
        assert released.resolved_at is not None

    def test_refund_on_failure(self):
        escrow = AgentEscrow()
        entry = escrow.hold_payment("ceo", "agent1", 0.05, "task-1")
        refunded = escrow.refund_on_failure(entry.escrow_id)
        assert refunded is not None
        assert refunded.status == "refunded"
        assert refunded.resolved_at is not None

    def test_release_already_released(self):
        escrow = AgentEscrow()
        entry = escrow.hold_payment("ceo", "agent1", 0.05, "task-1")
        escrow.release_on_completion(entry.escrow_id)
        assert escrow.release_on_completion(entry.escrow_id) is None

    def test_refund_already_refunded(self):
        escrow = AgentEscrow()
        entry = escrow.hold_payment("ceo", "agent1", 0.05, "task-1")
        escrow.refund_on_failure(entry.escrow_id)
        assert escrow.refund_on_failure(entry.escrow_id) is None

    def test_release_nonexistent(self):
        escrow = AgentEscrow()
        assert escrow.release_on_completion("fake") is None

    def test_refund_nonexistent(self):
        escrow = AgentEscrow()
        assert escrow.refund_on_failure("fake") is None

    def test_get_entry(self):
        escrow = AgentEscrow()
        entry = escrow.hold_payment("ceo", "agent1", 0.05, "task-1")
        assert escrow.get_entry(entry.escrow_id) is entry

    def test_get_entry_nonexistent(self):
        escrow = AgentEscrow()
        assert escrow.get_entry("nope") is None

    def test_get_entries_for_task(self):
        escrow = AgentEscrow()
        escrow.hold_payment("ceo", "agent1", 0.05, "task-1")
        escrow.hold_payment("ceo", "agent2", 0.10, "task-1")
        escrow.hold_payment("ceo", "agent3", 0.03, "task-2")
        assert len(escrow.get_entries_for_task("task-1")) == 2

    def test_list_held(self):
        escrow = AgentEscrow()
        e1 = escrow.hold_payment("ceo", "a", 0.01, "t1")
        e2 = escrow.hold_payment("ceo", "b", 0.02, "t2")
        escrow.release_on_completion(e1.escrow_id)
        held = escrow.list_held()
        assert len(held) == 1
        assert held[0].escrow_id == e2.escrow_id

    def test_list_all(self):
        escrow = AgentEscrow()
        escrow.hold_payment("ceo", "a", 0.01, "t1")
        escrow.hold_payment("ceo", "b", 0.02, "t2")
        assert len(escrow.list_all()) == 2

    def test_total_held(self):
        escrow = AgentEscrow()
        escrow.hold_payment("ceo", "a", 0.05, "t1")
        escrow.hold_payment("ceo", "b", 0.10, "t2")
        assert escrow.total_held() == pytest.approx(0.15)

    def test_total_released(self):
        escrow = AgentEscrow()
        e1 = escrow.hold_payment("ceo", "a", 0.05, "t1")
        escrow.hold_payment("ceo", "b", 0.10, "t2")
        escrow.release_on_completion(e1.escrow_id)
        assert escrow.total_released() == pytest.approx(0.05)
        assert escrow.total_held() == pytest.approx(0.10)

    def test_clear(self):
        escrow = AgentEscrow()
        escrow.hold_payment("ceo", "a", 0.05, "t1")
        escrow.clear()
        assert len(escrow.list_all()) == 0


# ===================================================================
# 8. BudgetTracker
# ===================================================================


class TestBudgetTracker:
    def test_initial_budget(self):
        bt = BudgetTracker(total_budget=50.0)
        assert bt.total_budget == 50.0
        assert bt.total_spent == 0.0
        assert bt.remaining == 50.0

    def test_can_afford(self):
        bt = BudgetTracker(total_budget=10.0)
        assert bt.can_afford(5.0) is True
        assert bt.can_afford(10.0) is True
        assert bt.can_afford(10.01) is False

    def test_spend(self):
        bt = BudgetTracker(total_budget=10.0)
        assert bt.spend("t1", 3.0) is True
        assert bt.total_spent == 3.0
        assert bt.remaining == 7.0

    def test_spend_exceeds_budget(self):
        bt = BudgetTracker(total_budget=5.0)
        assert bt.spend("t1", 3.0) is True
        assert bt.spend("t2", 3.0) is False  # only 2.0 remaining

    def test_get_spending(self):
        bt = BudgetTracker(total_budget=100.0)
        bt.spend("t1", 5.0)
        bt.spend("t1", 2.0)
        assert bt.get_spending("t1") == 7.0
        assert bt.get_spending("t2") == 0.0

    def test_spending_report(self):
        bt = BudgetTracker(total_budget=100.0)
        bt.spend("t1", 10.0)
        report = bt.spending_report()
        assert report["total_budget"] == 100.0
        assert report["total_spent"] == 10.0
        assert report["remaining"] == 90.0
        assert "t1" in report["tasks"]

    def test_reset(self):
        bt = BudgetTracker(total_budget=100.0)
        bt.spend("t1", 50.0)
        bt.reset()
        assert bt.total_spent == 0.0
        assert bt.remaining == 100.0


# ===================================================================
# 9. HiringManager — full lifecycle
# ===================================================================


class TestHiringManager:
    def _setup_manager(self) -> tuple[HiringManager, MarketplaceRegistry]:
        reg = _seeded_registry()
        escrow = AgentEscrow()
        budget = BudgetTracker(total_budget=1.0)
        manager = HiringManager(registry=reg, escrow=escrow, budget_tracker=budget)
        return manager, reg

    def test_discover(self):
        manager, _ = self._setup_manager()
        req = HireRequest(required_skills=["code"], budget=1.0)
        results = manager.discover(req)
        assert len(results) >= 1

    def test_select(self):
        manager, _ = self._setup_manager()
        req = HireRequest(required_skills=["code"], budget=1.0)
        candidates = manager.discover(req)
        agent = manager.select(candidates)
        assert agent is not None
        assert agent.name == "Coder"

    def test_select_empty(self):
        manager, _ = self._setup_manager()
        assert manager.select([]) is None

    def test_negotiate_price_ok(self):
        manager, _ = self._setup_manager()
        listing = _make_listing("Cheap", price=0.01)
        price = manager.negotiate_price(listing, 1.0)
        assert price == 0.01

    def test_negotiate_price_too_expensive(self):
        manager, _ = self._setup_manager()
        listing = _make_listing("Expensive", price=10.0)
        assert manager.negotiate_price(listing, 1.0) is None

    def test_hire_full_lifecycle(self):
        manager, reg = self._setup_manager()
        req = HireRequest(
            description="Write unit tests",
            required_skills=["code", "testing"],
            budget=1.0,
        )
        result = manager.hire(req)
        assert result.status == "completed"
        assert result.agent_name == "Coder"
        assert result.agreed_price == 0.02
        assert result.escrow_id != ""
        assert result.task_result is not None
        assert result.budget_remaining < 1.0

    def test_hire_no_matching_agents(self):
        reg = _fresh_registry()
        manager = HiringManager(registry=reg)
        req = HireRequest(
            description="Quantum computing",
            required_skills=["quantum"],
            budget=1.0,
        )
        result = manager.hire(req)
        assert result.status == "no_agents"

    def test_hire_budget_exceeded_per_agent(self):
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Expensive", ["code"], price=10.0))
        budget = BudgetTracker(total_budget=5.0)
        manager = HiringManager(registry=reg, budget_tracker=budget)
        req = HireRequest(
            description="Code",
            required_skills=["code"],
            budget=100.0,  # per-request budget high, but agent price > global budget
        )
        result = manager.hire(req)
        assert result.status == "budget_exceeded"

    def test_hire_budget_exhausted(self):
        reg = MarketplaceRegistry()
        reg.register_agent(_make_listing("Worker", ["code"], price=3.0))
        budget = BudgetTracker(total_budget=5.0)
        manager = HiringManager(registry=reg, budget_tracker=budget)

        # First hire works
        req1 = HireRequest(required_skills=["code"], budget=5.0)
        r1 = manager.hire(req1)
        assert r1.status == "completed"

        # Second hire — budget too low
        req2 = HireRequest(required_skills=["code"], budget=5.0)
        r2 = manager.hire(req2)
        assert r2.status == "budget_exceeded"

    def test_hire_history(self):
        manager, _ = self._setup_manager()
        req = HireRequest(required_skills=["code"], budget=1.0)
        manager.hire(req)
        assert len(manager.hire_history) == 1

    def test_escrow_released_on_success(self):
        manager, _ = self._setup_manager()
        req = HireRequest(required_skills=["code"], budget=1.0)
        result = manager.hire(req)
        entry = manager.escrow.get_entry(result.escrow_id)
        assert entry.status == "released"

    def test_jobs_increment_on_completion(self):
        manager, reg = self._setup_manager()
        coder = reg.get_agent_by_name("Coder")
        initial_jobs = coder.total_jobs
        req = HireRequest(required_skills=["code", "testing"], budget=1.0)
        manager.hire(req)
        assert coder.total_jobs == initial_jobs + 1


# ===================================================================
# 10. Marketplace API Endpoints
# ===================================================================


@pytest.fixture
def api_client():
    """Create a test client with marketplace routes mounted."""
    import src.api.marketplace_routes as routes_mod
    from fastapi import FastAPI

    # Create fresh state for each test
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

    # Patch module-level singletons
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

    # Restore originals
    routes_mod.marketplace = old_marketplace
    routes_mod._escrow = old_escrow
    routes_mod._budget = old_budget
    routes_mod._hiring_manager = old_hiring


class TestMarketplaceAPI:
    def test_list_agents(self, api_client):
        resp = api_client.get("/marketplace/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_list_agents_filter_skill(self, api_client):
        resp = api_client.get("/marketplace/agents?skill=code")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "APICoder"

    def test_list_agents_filter_max_price(self, api_client):
        resp = api_client.get("/marketplace/agents?max_price=0.02")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "APICoder"

    def test_get_agent_by_id(self, api_client):
        resp = api_client.get("/marketplace/agents/api-coder-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "APICoder"
        assert data["price_per_unit"] == 0.01

    def test_get_agent_not_found(self, api_client):
        resp = api_client.get("/marketplace/agents/nonexistent")
        assert resp.status_code == 404

    def test_hire_agent(self, api_client):
        resp = api_client.post("/marketplace/hire", json={
            "description": "Write some tests",
            "required_skills": ["code"],
            "budget": 1.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["agent_name"] == "APICoder"
        assert data["agreed_price"] == 0.01

    def test_hire_no_agents(self, api_client):
        resp = api_client.post("/marketplace/hire", json={
            "description": "Quantum research",
            "required_skills": ["quantum"],
            "budget": 1.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "no_agents"

    def test_list_jobs_empty(self, api_client):
        resp = api_client.get("/marketplace/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_jobs_after_hire(self, api_client):
        api_client.post("/marketplace/hire", json={
            "description": "Code task",
            "required_skills": ["code"],
            "budget": 1.0,
        })
        resp = api_client.get("/marketplace/jobs")
        assert resp.status_code == 200
        jobs = resp.json()
        assert len(jobs) == 1
        assert jobs[0]["status"] == "completed"
        assert jobs[0]["escrow_status"] == "released"

    def test_budget_endpoint(self, api_client):
        resp = api_client.get("/marketplace/budget")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_budget" in data
        assert "remaining" in data

    def test_x402_info_endpoint(self, api_client):
        resp = api_client.get("/marketplace/x402")
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] == "Payment Required"
        assert "accepts" in data
        assert len(data["accepts"]) == 1
        accept = data["accepts"][0]
        assert accept["scheme"] == "exact"
        assert accept["network"] == "eip155:8453"
        assert accept["mimeType"] == "application/json"
