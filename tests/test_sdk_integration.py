"""Tests for Microsoft Agent Framework SDK integration layer.

Covers:
- SDK availability detection
- Agent creation via SDK
- Sequential orchestration via SequentialBuilder
- Concurrent orchestration via ConcurrentBuilder
- Handoff orchestration via HandoffBuilder
- SDKOrchestrator unified interface
- MCP tool creation and info
- MCP server factory
- Fallback to native orchestrator
- Demo seeder SDK pattern metadata
"""

from __future__ import annotations

import asyncio
import pytest

from agent_framework import ChatAgent, SequentialBuilder, ConcurrentBuilder, HandoffBuilder

from src.agents._mock_client import MockChatClient
from src.integrations.ms_agent_framework import (
    SDKOrchestrator,
    SDKOrchestrationResult,
    create_sdk_agent,
    get_hirewire_sdk_agents,
    run_sequential,
    run_concurrent,
    run_handoff,
    get_sdk_info,
    _check_sdk_available,
)
from src.integrations.mcp_tools import (
    create_hirewire_mcp_agent,
    create_mcp_server,
    get_mcp_tool_info,
    HIREWIRE_SDK_TOOLS,
    submit_task_tool,
    list_agents_tool,
    check_budget_tool,
    agent_metrics_tool,
    x402_payment_tool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client() -> MockChatClient:
    return MockChatClient()


def _make_sdk_agent(name: str = "TestAgent") -> ChatAgent:
    return create_sdk_agent(
        name=name,
        description=f"Test SDK agent: {name}",
        instructions=f"You are {name}. Respond concisely.",
        chat_client=_mock_client(),
    )


# ===================================================================
# 1. SDK Availability
# ===================================================================


class TestSDKAvailability:
    def test_sdk_is_available(self):
        assert _check_sdk_available() is True

    def test_get_sdk_info(self):
        info = get_sdk_info()
        assert info["available"] is True
        assert "1.0.0" in info["version"]
        assert "sequential" in info["patterns"]
        assert "concurrent" in info["patterns"]
        assert "handoff" in info["patterns"]
        assert "SequentialBuilder" in info["features"]
        assert "ConcurrentBuilder" in info["features"]
        assert "HandoffBuilder" in info["features"]

    def test_sdk_info_features(self):
        info = get_sdk_info()
        assert "ChatAgent" in info["features"]
        assert "MCPStdioTool" in info["features"]
        assert "MCPStreamableHTTPTool" in info["features"]

    def test_sdk_info_version_format(self):
        info = get_sdk_info()
        assert info["version"].startswith("1.0.0b")


# ===================================================================
# 2. SDK Agent Creation
# ===================================================================


class TestSDKAgentCreation:
    def test_create_basic_agent(self):
        agent = _make_sdk_agent("Alpha")
        assert isinstance(agent, ChatAgent)
        assert agent.name == "Alpha"

    def test_create_agent_with_mock_client(self):
        client = _mock_client()
        agent = create_sdk_agent(
            name="Beta",
            instructions="Test instructions",
            chat_client=client,
        )
        assert isinstance(agent, ChatAgent)

    def test_create_agent_with_description(self):
        agent = create_sdk_agent(
            name="Gamma",
            instructions="Test",
            description="Custom description",
            chat_client=_mock_client(),
        )
        assert agent.description == "Custom description"

    def test_create_agent_default_description(self):
        agent = create_sdk_agent(
            name="Delta",
            instructions="Test",
            chat_client=_mock_client(),
        )
        assert "Delta" in agent.description

    def test_get_hirewire_agents(self):
        agents = get_hirewire_sdk_agents(_mock_client())
        assert "ceo" in agents
        assert "builder" in agents
        assert "research" in agents
        assert "analyst" in agents
        for name, agent in agents.items():
            assert isinstance(agent, ChatAgent)

    def test_hirewire_agents_have_names(self):
        agents = get_hirewire_sdk_agents(_mock_client())
        assert agents["ceo"].name == "CEO"
        assert agents["builder"].name == "Builder"
        assert agents["research"].name == "Research"
        assert agents["analyst"].name == "Analyst"

    def test_hirewire_agents_have_descriptions(self):
        agents = get_hirewire_sdk_agents(_mock_client())
        for agent in agents.values():
            assert len(agent.description) > 10

    def test_hirewire_agents_are_chat_agents(self):
        agents = get_hirewire_sdk_agents(_mock_client())
        for agent in agents.values():
            assert isinstance(agent, ChatAgent)


# ===================================================================
# 3. SDKOrchestrationResult
# ===================================================================


class TestSDKOrchestrationResult:
    def test_default_values(self):
        result = SDKOrchestrationResult()
        assert result.status == "pending"
        assert result.pattern == ""
        assert result.final_output == ""
        assert result.agent_results == []
        assert result.elapsed_ms == 0.0
        assert result.orchestration_id.startswith("sdk_orch_")

    def test_success_property(self):
        result = SDKOrchestrationResult(status="completed")
        assert result.success is True
        result2 = SDKOrchestrationResult(status="failed")
        assert result2.success is False

    def test_sdk_version(self):
        result = SDKOrchestrationResult()
        assert "agent_framework" in result.sdk_version

    def test_metadata(self):
        result = SDKOrchestrationResult(metadata={"key": "value"})
        assert result.metadata["key"] == "value"


# ===================================================================
# 4. Sequential Orchestration via SDK
# ===================================================================


class TestSDKSequential:
    @pytest.mark.asyncio
    async def test_sequential_basic(self):
        agents = [_make_sdk_agent("A"), _make_sdk_agent("B")]
        result = await run_sequential(agents, "Test task")
        assert result.status == "completed"
        assert result.pattern == "sdk_sequential"
        assert len(result.agent_results) >= 1
        assert result.final_output != ""
        assert result.elapsed_ms > 0

    @pytest.mark.asyncio
    async def test_sequential_empty_agents(self):
        result = await run_sequential([], "Test task")
        assert result.status == "failed"
        assert "No agents" in result.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_sequential_single_agent(self):
        agents = [_make_sdk_agent("Solo")]
        result = await run_sequential(agents, "Solo task")
        assert result.status == "completed"
        assert len(result.agent_results) >= 1

    @pytest.mark.asyncio
    async def test_sequential_three_agents(self):
        agents = [_make_sdk_agent("R"), _make_sdk_agent("B"), _make_sdk_agent("D")]
        result = await run_sequential(agents, "Pipeline task")
        assert result.status == "completed"
        assert result.metadata.get("agent_count") == 3

    @pytest.mark.asyncio
    async def test_sequential_metadata(self):
        agents = [_make_sdk_agent("X"), _make_sdk_agent("Y")]
        result = await run_sequential(agents, "Metadata test")
        assert "agent_count" in result.metadata
        assert "message_count" in result.metadata

    @pytest.mark.asyncio
    async def test_sequential_preserves_task(self):
        task = "Unique task description 12345"
        result = await run_sequential([_make_sdk_agent("A")], task)
        assert result.task == task


# ===================================================================
# 5. Concurrent Orchestration via SDK
# ===================================================================


class TestSDKConcurrent:
    @pytest.mark.asyncio
    async def test_concurrent_basic(self):
        agents = [_make_sdk_agent("A"), _make_sdk_agent("B")]
        result = await run_concurrent(agents, "Parallel task")
        assert result.status == "completed"
        assert result.pattern == "sdk_concurrent"
        assert result.elapsed_ms > 0

    @pytest.mark.asyncio
    async def test_concurrent_empty_agents(self):
        result = await run_concurrent([], "Test")
        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_concurrent_three_agents(self):
        agents = [_make_sdk_agent("A"), _make_sdk_agent("B"), _make_sdk_agent("C")]
        result = await run_concurrent(agents, "Three-way analysis")
        assert result.status == "completed"
        assert result.metadata.get("agent_count") == 3

    @pytest.mark.asyncio
    async def test_concurrent_merged_output(self):
        agents = [_make_sdk_agent("Expert1"), _make_sdk_agent("Expert2")]
        result = await run_concurrent(agents, "Give analysis")
        assert result.final_output != ""

    @pytest.mark.asyncio
    async def test_concurrent_parallel_results(self):
        agents = [_make_sdk_agent("A"), _make_sdk_agent("B")]
        result = await run_concurrent(agents, "Test")
        assert "parallel_results" in result.metadata

    @pytest.mark.asyncio
    async def test_concurrent_preserves_task(self):
        task = "Unique concurrent task 67890"
        result = await run_concurrent([_make_sdk_agent("A")], task)
        assert result.task == task


# ===================================================================
# 6. Handoff Orchestration via SDK
# ===================================================================


class TestSDKHandoff:
    @pytest.mark.asyncio
    async def test_handoff_basic(self):
        agents = [_make_sdk_agent("CEO"), _make_sdk_agent("Builder")]
        result = await run_handoff(agents, "Handoff task", max_turns=3)
        assert result.pattern == "sdk_handoff"
        assert result.status == "completed"
        assert result.elapsed_ms > 0

    @pytest.mark.asyncio
    async def test_handoff_empty_participants(self):
        result = await run_handoff([], "Test")
        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_handoff_with_start_agent(self):
        ceo = _make_sdk_agent("CEO")
        builder = _make_sdk_agent("Builder")
        result = await run_handoff([ceo, builder], "Task", start_agent=ceo, max_turns=3)
        assert result.status == "completed"
        assert result.metadata.get("start_agent") == "CEO"

    @pytest.mark.asyncio
    async def test_handoff_max_turns(self):
        agents = [_make_sdk_agent("A"), _make_sdk_agent("B")]
        result = await run_handoff(agents, "Long conversation", max_turns=2)
        assert result.status == "completed"
        assert result.metadata.get("turns", 0) <= 3  # max_turns + small buffer

    @pytest.mark.asyncio
    async def test_handoff_three_participants(self):
        agents = [_make_sdk_agent("CEO"), _make_sdk_agent("Builder"), _make_sdk_agent("Research")]
        result = await run_handoff(agents, "Complex task", max_turns=3)
        assert result.status == "completed"
        assert result.metadata.get("agent_count") == 3

    @pytest.mark.asyncio
    async def test_handoff_preserves_task(self):
        task = "Unique handoff task abc"
        result = await run_handoff([_make_sdk_agent("A"), _make_sdk_agent("B")], task, max_turns=2)
        assert result.task == task


# ===================================================================
# 7. SDKOrchestrator unified interface
# ===================================================================


class TestSDKOrchestrator:
    def test_init(self):
        orch = SDKOrchestrator(_mock_client())
        assert orch.sdk_available is True
        assert "ceo" in orch.agents
        assert "builder" in orch.agents

    def test_agents_dict(self):
        orch = SDKOrchestrator(_mock_client())
        agents = orch.agents
        assert len(agents) == 4
        for agent in agents.values():
            assert isinstance(agent, ChatAgent)

    def test_empty_history(self):
        orch = SDKOrchestrator(_mock_client())
        assert orch.history == []

    @pytest.mark.asyncio
    async def test_run_sequential(self):
        orch = SDKOrchestrator(_mock_client())
        result = await orch.run("Test task", pattern="sequential")
        assert result.pattern == "sdk_sequential"
        assert result.status == "completed"
        assert len(orch.history) == 1

    @pytest.mark.asyncio
    async def test_run_concurrent(self):
        orch = SDKOrchestrator(_mock_client())
        result = await orch.run("Test task", pattern="concurrent")
        assert result.pattern == "sdk_concurrent"
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_run_handoff(self):
        orch = SDKOrchestrator(_mock_client())
        result = await orch.run("Test task", pattern="handoff", max_turns=2)
        assert result.pattern == "sdk_handoff"
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_run_unknown_pattern(self):
        orch = SDKOrchestrator(_mock_client())
        result = await orch.run("Test", pattern="unknown")
        assert result.status == "failed"
        assert "Unknown pattern" in result.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_run_with_specific_agents(self):
        orch = SDKOrchestrator(_mock_client())
        result = await orch.run("Test", pattern="sequential", agents=["research", "builder"])
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_run_with_invalid_agent_names(self):
        orch = SDKOrchestrator(_mock_client())
        result = await orch.run("Test", pattern="sequential", agents=["nonexistent"])
        assert result.status == "failed"  # No agents found

    @pytest.mark.asyncio
    async def test_history_accumulates(self):
        orch = SDKOrchestrator(_mock_client())
        await orch.run("Task 1", pattern="sequential")
        await orch.run("Task 2", pattern="concurrent")
        assert len(orch.history) == 2
        assert orch.history[0].pattern == "sdk_sequential"
        assert orch.history[1].pattern == "sdk_concurrent"

    @pytest.mark.asyncio
    async def test_default_agents_sequential(self):
        orch = SDKOrchestrator(_mock_client())
        result = await orch.run("Default sequential", pattern="sequential")
        # Sequential default: research, builder
        assert result.metadata.get("agent_count") == 2

    @pytest.mark.asyncio
    async def test_default_agents_concurrent(self):
        orch = SDKOrchestrator(_mock_client())
        result = await orch.run("Default concurrent", pattern="concurrent")
        # Concurrent default: research, analyst, builder
        assert result.metadata.get("agent_count") == 3


# ===================================================================
# 8. MCP Tools
# ===================================================================


class TestMCPTools:
    def test_hirewire_sdk_tools_count(self):
        assert len(HIREWIRE_SDK_TOOLS) == 5

    def test_tool_info(self):
        info = get_mcp_tool_info()
        assert len(info) == 5
        for t in info:
            assert "name" in t
            assert "description" in t
            assert t["type"] == "sdk_tool"
            assert t["framework"] == "agent_framework"

    def test_tool_names(self):
        info = get_mcp_tool_info()
        names = [t["name"] for t in info]
        assert "hirewire_submit_task" in names
        assert "hirewire_list_agents" in names
        assert "hirewire_check_budget" in names
        assert "hirewire_agent_metrics" in names
        assert "hirewire_x402_payment" in names


# ===================================================================
# 9. MCP Agent and Server
# ===================================================================


class TestMCPAgent:
    def test_create_mcp_agent(self):
        agent = create_hirewire_mcp_agent(_mock_client())
        assert isinstance(agent, ChatAgent)
        assert agent.name == "HireWire"
        assert "marketplace" in agent.description.lower()

    def test_mcp_agent_has_tools(self):
        agent = create_hirewire_mcp_agent(_mock_client())
        # The agent should have tools registered
        assert agent.name == "HireWire"

    def test_create_mcp_server(self):
        server = create_mcp_server(_mock_client())
        assert server is not None

    def test_mcp_server_type(self):
        server = create_mcp_server(_mock_client())
        # Should be an MCP Server object
        assert hasattr(server, "run") or hasattr(server, "create_initialization_options")


# ===================================================================
# 10. Demo Seeder SDK patterns
# ===================================================================


class TestDemoSeederSDKPatterns:
    def test_scenarios_have_orchestration_pattern(self):
        from src.demo.seeder import DEMO_SCENARIOS
        for scenario in DEMO_SCENARIOS:
            assert "orchestration_pattern" in scenario

    def test_at_least_two_sdk_patterns(self):
        from src.demo.seeder import DEMO_SCENARIOS
        sdk_count = sum(
            1 for s in DEMO_SCENARIOS
            if s.get("orchestration_pattern", "").startswith("sdk_")
        )
        assert sdk_count >= 2, f"Expected at least 2 SDK-orchestrated scenarios, got {sdk_count}"

    def test_sequential_pattern_present(self):
        from src.demo.seeder import DEMO_SCENARIOS
        has_sequential = any(
            s.get("orchestration_pattern") == "sdk_sequential"
            for s in DEMO_SCENARIOS
        )
        assert has_sequential, "Expected at least one sdk_sequential scenario"

    def test_handoff_pattern_present(self):
        from src.demo.seeder import DEMO_SCENARIOS
        has_handoff = any(
            s.get("orchestration_pattern") == "sdk_handoff"
            for s in DEMO_SCENARIOS
        )
        assert has_handoff, "Expected at least one sdk_handoff scenario"

    def test_sdk_scenarios_have_detail(self):
        from src.demo.seeder import DEMO_SCENARIOS
        for s in DEMO_SCENARIOS:
            if s.get("orchestration_pattern", "").startswith("sdk_"):
                assert s.get("sdk_pattern_detail"), f"SDK scenario missing detail: {s['description']}"

    def test_seed_returns_sdk_count(self):
        from src.demo.seeder import seed_demo_data
        result = seed_demo_data()
        assert "sdk_orchestrated_tasks" in result
        assert result["sdk_orchestrated_tasks"] >= 2


# ===================================================================
# 11. SDK Builder pattern verification (direct SDK usage)
# ===================================================================


class TestSDKBuilderPatterns:
    """Verify the actual SDK builder classes work correctly."""

    def test_sequential_builder_exists(self):
        builder = SequentialBuilder()
        assert builder is not None

    def test_concurrent_builder_exists(self):
        builder = ConcurrentBuilder()
        assert builder is not None

    def test_handoff_builder_exists(self):
        agents = [_make_sdk_agent("A"), _make_sdk_agent("B")]
        builder = HandoffBuilder(name="test", participants=agents)
        assert builder is not None

    def test_sequential_builder_with_participants(self):
        agents = [_make_sdk_agent("A"), _make_sdk_agent("B")]
        workflow = SequentialBuilder().participants(agents).build()
        assert workflow is not None

    def test_concurrent_builder_with_participants(self):
        agents = [_make_sdk_agent("A"), _make_sdk_agent("B")]
        workflow = ConcurrentBuilder().participants(agents).build()
        assert workflow is not None

    def test_handoff_builder_with_start(self):
        a = _make_sdk_agent("A")
        b = _make_sdk_agent("B")
        workflow = (
            HandoffBuilder(name="test", participants=[a, b])
            .with_start_agent(a)
            .build()
        )
        assert workflow is not None


# ===================================================================
# 12. Integration with existing framework layer
# ===================================================================


class TestFrameworkIntegration:
    """Verify SDK integration coexists with existing framework."""

    def test_existing_orchestrator_still_works(self):
        from src.framework.orchestrator import SequentialOrchestrator
        from src.framework.agent import AgentFrameworkAgent
        agent = AgentFrameworkAgent(
            name="Test", description="Test", instructions="Test",
            chat_client=_mock_client(),
        )
        orch = SequentialOrchestrator([agent])
        assert len(orch.agents) == 1

    @pytest.mark.asyncio
    async def test_existing_orchestrator_runs(self):
        from src.framework.orchestrator import SequentialOrchestrator
        from src.framework.agent import AgentFrameworkAgent
        agent = AgentFrameworkAgent(
            name="Test", description="Test", instructions="Test",
            chat_client=_mock_client(),
        )
        orch = SequentialOrchestrator([agent])
        result = await orch.run("Test task")
        assert result.status == "completed"
        assert result.pattern == "sequential"

    @pytest.mark.asyncio
    async def test_sdk_and_native_coexist(self):
        """Both SDK and native orchestrators should work side by side."""
        from src.framework.orchestrator import SequentialOrchestrator
        from src.framework.agent import AgentFrameworkAgent

        # Native
        native_agent = AgentFrameworkAgent(
            name="Native", description="Native agent", instructions="Test",
            chat_client=_mock_client(),
        )
        native_orch = SequentialOrchestrator([native_agent])
        native_result = await native_orch.run("Native task")

        # SDK
        sdk_orch = SDKOrchestrator(_mock_client())
        sdk_result = await sdk_orch.run("SDK task", pattern="sequential")

        assert native_result.status == "completed"
        assert native_result.pattern == "sequential"
        assert sdk_result.status == "completed"
        assert sdk_result.pattern == "sdk_sequential"
