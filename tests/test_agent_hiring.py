"""Tests for external agent hiring feature.

Covers:
- Mock external agent endpoints
- Agent discovery (external agents)
- Hiring workflow end-to-end
- Budget deduction after hiring
- Demo scenario completion
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import pytest_asyncio
import uvicorn

from src.external.mock_agent import (
    create_mock_agent_app,
    clear_processed_tasks,
    get_processed_tasks,
    AGENT_CARD,
)
from src.mcp_servers.payment_hub import ledger
from src.mcp_servers.registry_server import registry
from src.workflows.hiring import (
    discover_external_agents,
    evaluate_agent,
    run_hiring_workflow,
    HiringDecision,
)


DESIGNER_PORT = 9111  # Use a different port from the demo to avoid conflicts


@pytest.fixture(autouse=True)
def _reset_ledger():
    """Clear ledger state between tests."""
    ledger._transactions.clear()
    ledger._budgets.clear()
    ledger._tx_counter = 0
    yield
    ledger._transactions.clear()
    ledger._budgets.clear()
    ledger._tx_counter = 0


@pytest.fixture(autouse=True)
def _reset_mock_agent():
    """Clear mock agent processed tasks between tests."""
    clear_processed_tasks()
    yield
    clear_processed_tasks()


@pytest_asyncio.fixture()
async def designer_server():
    """Start mock designer agent as a background server for the test."""
    app = create_mock_agent_app(port=DESIGNER_PORT)
    config = uvicorn.Config(app, host="127.0.0.1", port=DESIGNER_PORT, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # Wait until server is ready
    for _ in range(50):
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"http://127.0.0.1:{DESIGNER_PORT}/health")
                if r.status_code == 200:
                    break
        except httpx.ConnectError:
            pass
        await asyncio.sleep(0.1)
    yield server
    server.should_exit = True
    await task


# ---------------------------------------------------------------------------
# Mock External Agent Tests
# ---------------------------------------------------------------------------

class TestMockExternalAgent:
    """Test that the mock external designer agent works correctly."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, designer_server):
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{DESIGNER_PORT}/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_agent_card_endpoint(self, designer_server):
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{DESIGNER_PORT}/agent-card")
        assert resp.status_code == 200
        card = resp.json()
        assert card["name"] == "designer-ext-001"
        assert "design" in card["skills"]
        assert card["pricing"]["amount_usdc"] == 0.05

    @pytest.mark.asyncio
    async def test_task_submission(self, designer_server):
        payload = {
            "task_id": "test-001",
            "description": "Design a landing page",
            "from_agent": "ceo",
            "budget": 1.0,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{DESIGNER_PORT}/a2a/tasks", json=payload,
            )
        assert resp.status_code == 200
        result = resp.json()
        assert result["status"] == "completed"
        assert result["task_id"] == "test-001"
        assert result["agent"] == "designer-ext-001"
        assert result["price_usdc"] == 0.05
        assert "Design Deliverable" in result["deliverable"]

    @pytest.mark.asyncio
    async def test_landing_page_deliverable(self, designer_server):
        payload = {
            "task_id": "test-lp",
            "description": "Create a beautiful landing page design",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{DESIGNER_PORT}/a2a/tasks", json=payload,
            )
        result = resp.json()
        assert "Landing Page" in result["deliverable"]
        assert "Color Palette" in result["deliverable"]

    @pytest.mark.asyncio
    async def test_processed_tasks_tracked(self, designer_server):
        payload = {"task_id": "track-001", "description": "Generic design task"}
        async with httpx.AsyncClient() as client:
            await client.post(
                f"http://127.0.0.1:{DESIGNER_PORT}/a2a/tasks", json=payload,
            )
        tasks = get_processed_tasks()
        assert len(tasks) == 1
        assert tasks[0].task_id == "track-001"


# ---------------------------------------------------------------------------
# Agent Discovery Tests
# ---------------------------------------------------------------------------

class TestAgentDiscovery:
    """Test that external agent discovery works via the registry."""

    def test_discover_external_design_agents(self):
        agents = discover_external_agents("design")
        assert len(agents) >= 1
        assert any(a.name == "designer-ext-001" for a in agents)

    def test_discover_filters_internal_agents(self):
        agents = discover_external_agents("code")
        # builder is internal, should not appear
        assert all(a.is_external for a in agents)

    def test_discover_with_max_price(self):
        agents = discover_external_agents("design", max_price=0.10)
        assert len(agents) >= 1  # designer costs $0.05

    def test_discover_with_low_price_limit(self):
        agents = discover_external_agents("design", max_price=0.01)
        assert len(agents) == 0  # designer costs $0.05, should be filtered

    def test_registry_has_external_agent(self):
        agent = registry.get("designer-ext-001")
        assert agent is not None
        assert agent.is_external is True
        assert agent.protocol == "a2a"
        assert agent.payment == "x402"


# ---------------------------------------------------------------------------
# Agent Evaluation Tests
# ---------------------------------------------------------------------------

class TestAgentEvaluation:
    def test_evaluate_good_match(self):
        agent = registry.get("designer-ext-001")
        decision = evaluate_agent(agent, ["design", "ui", "landing-page"])
        assert decision.approved is True
        assert decision.capability_match == 1.0  # all 3 skills match

    def test_evaluate_partial_match(self):
        agent = registry.get("designer-ext-001")
        decision = evaluate_agent(agent, ["design", "video-editing"])
        assert decision.approved is True
        assert 0.3 <= decision.capability_match < 1.0

    def test_evaluate_no_match(self):
        agent = registry.get("designer-ext-001")
        decision = evaluate_agent(agent, ["kubernetes", "devops", "terraform"])
        assert decision.approved is False
        assert decision.capability_match < 0.3


# ---------------------------------------------------------------------------
# Hiring Workflow Tests
# ---------------------------------------------------------------------------

class TestHiringWorkflow:
    """Test the full hiring workflow end-to-end."""

    @pytest.mark.asyncio
    async def test_hiring_completes(self, designer_server):
        # Update endpoint to match test server port
        agent = registry.get("designer-ext-001")
        original_endpoint = agent.endpoint
        agent.endpoint = f"http://127.0.0.1:{DESIGNER_PORT}"
        try:
            result = await run_hiring_workflow(
                task_id="hire-test-001",
                task_description="Design a landing page for AgentOS",
                required_skills=["design", "ui"],
                budget_usd=1.0,
                capability_query="design",
            )
            assert result.status == "completed"
            assert result.external_result is not None
            assert result.payment is not None
        finally:
            agent.endpoint = original_endpoint

    @pytest.mark.asyncio
    async def test_budget_deducted_after_hiring(self, designer_server):
        agent = registry.get("designer-ext-001")
        original_endpoint = agent.endpoint
        agent.endpoint = f"http://127.0.0.1:{DESIGNER_PORT}"
        try:
            result = await run_hiring_workflow(
                task_id="budget-test-001",
                task_description="Design a logo",
                required_skills=["design"],
                budget_usd=1.0,
            )
            assert result.status == "completed"
            budget = ledger.get_budget("budget-test-001")
            assert budget.spent == 0.05  # designer charges $0.05
            assert budget.remaining == 0.95
        finally:
            agent.endpoint = original_endpoint

    @pytest.mark.asyncio
    async def test_hiring_no_agents_found(self):
        result = await run_hiring_workflow(
            task_id="no-agent-001",
            task_description="Do quantum computing",
            required_skills=["quantum"],
            budget_usd=1.0,
            capability_query="quantum-computing-xyz",
        )
        assert result.status == "no_agents_found"
        assert result.external_result is None
        assert result.payment is None

    @pytest.mark.asyncio
    async def test_hiring_records_payment(self, designer_server):
        agent = registry.get("designer-ext-001")
        original_endpoint = agent.endpoint
        agent.endpoint = f"http://127.0.0.1:{DESIGNER_PORT}"
        try:
            result = await run_hiring_workflow(
                task_id="payment-test-001",
                task_description="Design mockups",
                required_skills=["design"],
                budget_usd=2.0,
            )
            assert result.payment is not None
            assert result.payment["tx_id"].startswith("tx_")
            assert result.payment["amount_usdc"] == 0.05
            assert result.payment["to_agent"] == "designer-ext-001"

            # Verify in the ledger
            txs = ledger.get_transactions("payment-test-001")
            assert len(txs) == 1
            assert txs[0].to_agent == "designer-ext-001"
        finally:
            agent.endpoint = original_endpoint

    @pytest.mark.asyncio
    async def test_hiring_discovery_returns_agents(self, designer_server):
        agent = registry.get("designer-ext-001")
        original_endpoint = agent.endpoint
        agent.endpoint = f"http://127.0.0.1:{DESIGNER_PORT}"
        try:
            result = await run_hiring_workflow(
                task_id="disc-test-001",
                task_description="Design a page",
                required_skills=["design"],
                budget_usd=1.0,
            )
            assert len(result.discovery) >= 1
            assert result.discovery[0]["name"] == "designer-ext-001"
        finally:
            agent.endpoint = original_endpoint


# ---------------------------------------------------------------------------
# Demo Scenario Test
# ---------------------------------------------------------------------------

class TestAgentHiringDemoScenario:
    @pytest.mark.asyncio
    async def test_demo_scenario_completes(self, designer_server):
        """Run the full demo scenario end-to-end.

        Note: The demo creates its own server on port 9100, so we need to
        update the registry endpoint to match the test server port instead.
        """
        agent = registry.get("designer-ext-001")
        original_endpoint = agent.endpoint
        agent.endpoint = f"http://127.0.0.1:{DESIGNER_PORT}"
        try:
            # Import and run just the hiring workflow part
            result = await run_hiring_workflow(
                task_id="demo-hiring-test",
                task_description="Create a professional design for an AgentOS landing page",
                required_skills=["design", "ui", "landing-page"],
                budget_usd=5.0,
                capability_query="design",
            )
            assert result.status == "completed"
            assert result.external_result is not None
            assert "deliverable" in result.external_result
            assert result.payment is not None
            assert result.budget_summary["spent"] == 0.05
            assert result.budget_summary["remaining"] == 4.95
        finally:
            agent.endpoint = original_endpoint
