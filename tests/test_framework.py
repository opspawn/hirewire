"""Tests for Microsoft Agent Framework integration layer.

Covers:
- AgentFrameworkAgent creation, tool management, threads, invocation
- Orchestrator patterns: sequential, concurrent, handoff
- MCP tool registration, invocation, and search
- A2A agent card generation and server dispatch
- Pre-built agents: Researcher, Analyst, Executor
"""

from __future__ import annotations

import asyncio
import pytest

from src.agents._mock_client import MockChatClient
from src.framework.agent import (
    AgentFrameworkAgent,
    AgentThread,
    AgentMessage,
    ToolDescriptor,
)
from src.framework.orchestrator import (
    SequentialOrchestrator,
    ConcurrentOrchestrator,
    HandoffOrchestrator,
    OrchestratorResult,
)
from src.framework.mcp_tools import (
    MCPToolDescriptor,
    MCPToolRegistry,
    SCREENSHOT_TOOL,
    AI_ANALYSIS_TOOL,
    MARKDOWN_CONVERSION_TOOL,
    WEB_SEARCH_TOOL,
    FILE_OPERATION_TOOL,
    API_CALL_TOOL,
    BUILTIN_TOOLS,
    create_default_registry,
)
from src.framework.a2a import A2AAgentCard, A2AServer, A2AClient, A2ATask
from src.framework.agents.researcher import create_researcher_agent
from src.framework.agents.analyst import create_analyst_agent
from src.framework.agents.executor import create_executor_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client() -> MockChatClient:
    return MockChatClient()


def _make_agent(name: str = "TestAgent", tools: list | None = None) -> AgentFrameworkAgent:
    return AgentFrameworkAgent(
        name=name,
        description=f"Test agent: {name}",
        instructions=f"You are {name}.",
        tools=tools or [],
        chat_client=_mock_client(),
    )


async def _echo_handler(args: dict) -> dict:
    """Simple echo handler for testing."""
    return {"echoed": args}


async def _fail_handler(args: dict) -> dict:
    """Handler that always raises."""
    raise RuntimeError("intentional failure")


def _make_tool(name: str = "test_tool", handler=None) -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        description=f"Test tool: {name}",
        parameters={"type": "object", "properties": {}},
        execute=handler or _echo_handler,
    )


# ===================================================================
# 1. AgentFrameworkAgent — creation and metadata
# ===================================================================


class TestAgentCreation:
    def test_create_basic_agent(self):
        agent = _make_agent("Alpha")
        assert agent.name == "Alpha"
        assert "Alpha" in agent.description
        assert agent.invoke_count == 0

    def test_agent_repr(self):
        agent = _make_agent("Beta")
        r = repr(agent)
        assert "Beta" in r
        assert "tools=0" in r

    def test_agent_card_property(self):
        agent = _make_agent("Gamma")
        card = agent.agent_card
        assert card["name"] == "Gamma"
        assert card["capabilities"]["invoke"] is True
        assert card["capabilities"]["threads"] is True
        assert card["capabilities"]["delegate"] is False

    def test_agent_with_tools(self):
        tool = _make_tool("calc")
        agent = _make_agent("Delta", tools=[tool])
        assert len(agent.list_tools()) == 1
        assert agent.get_tool("calc") is not None
        assert agent.get_tool("nonexistent") is None


# ===================================================================
# 2. AgentFrameworkAgent — tool management
# ===================================================================


class TestToolManagement:
    def test_add_tool(self):
        agent = _make_agent()
        assert len(agent.list_tools()) == 0
        agent.add_tool(_make_tool("t1"))
        assert len(agent.list_tools()) == 1

    def test_remove_tool(self):
        agent = _make_agent()
        agent.add_tool(_make_tool("t1"))
        assert agent.remove_tool("t1") is True
        assert len(agent.list_tools()) == 0

    def test_remove_nonexistent_tool(self):
        agent = _make_agent()
        assert agent.remove_tool("nope") is False

    def test_list_tools_empty(self):
        agent = _make_agent()
        assert agent.list_tools() == []


# ===================================================================
# 3. AgentFrameworkAgent — connected agents
# ===================================================================


class TestConnectedAgents:
    def test_connect_agent(self):
        primary = _make_agent("Primary")
        specialist = _make_agent("Specialist")
        primary.connect_agent(specialist)
        assert len(primary.get_connected_agents()) == 1
        assert primary.get_connected_agent("Specialist") is specialist

    def test_disconnect_agent(self):
        primary = _make_agent("Primary")
        specialist = _make_agent("Specialist")
        primary.connect_agent(specialist)
        assert primary.disconnect_agent("Specialist") is True
        assert len(primary.get_connected_agents()) == 0

    def test_disconnect_nonexistent(self):
        agent = _make_agent()
        assert agent.disconnect_agent("nope") is False

    def test_agent_card_with_connected(self):
        primary = _make_agent("P")
        s1 = _make_agent("S1")
        s2 = _make_agent("S2")
        primary.connect_agent(s1)
        primary.connect_agent(s2)
        card = primary.agent_card
        assert len(card["connected_agents"]) == 2
        assert card["capabilities"]["delegate"] is True


# ===================================================================
# 4. AgentFrameworkAgent — thread management
# ===================================================================


class TestThreadManagement:
    def test_create_thread(self):
        agent = _make_agent()
        thread = agent.create_thread()
        assert thread.thread_id.startswith("thread_")
        assert len(thread.messages) == 0

    def test_get_thread(self):
        agent = _make_agent()
        thread = agent.create_thread()
        found = agent.get_thread(thread.thread_id)
        assert found is thread

    def test_get_thread_nonexistent(self):
        agent = _make_agent()
        assert agent.get_thread("nonexistent") is None

    def test_list_threads(self):
        agent = _make_agent()
        agent.create_thread()
        agent.create_thread()
        assert len(agent.list_threads()) == 2

    def test_thread_messages(self):
        thread = AgentThread()
        thread.add_message("user", "Hello")
        thread.add_message("assistant", "Hi there", agent_name="Bot")
        assert len(thread.messages) == 2
        assert thread.messages[0].role == "user"
        assert thread.messages[1].agent_name == "Bot"

    def test_thread_get_history_limit(self):
        thread = AgentThread()
        for i in range(10):
            thread.add_message("user", f"msg {i}")
        history = thread.get_history(max_messages=3)
        assert len(history) == 3
        assert history[0].content == "msg 7"

    def test_thread_clear(self):
        thread = AgentThread()
        thread.add_message("user", "test")
        thread.clear()
        assert len(thread.messages) == 0


# ===================================================================
# 5. AgentFrameworkAgent — invocation
# ===================================================================


class TestAgentInvocation:
    @pytest.mark.asyncio
    async def test_invoke_basic(self):
        agent = _make_agent("Invoker")
        result = await agent.invoke("What is 2+2?")
        assert result["agent"] == "Invoker"
        assert "response" in result
        assert result["invoke_count"] == 1
        assert result["thread_id"].startswith("thread_")

    @pytest.mark.asyncio
    async def test_invoke_increments_count(self):
        agent = _make_agent("Counter")
        await agent.invoke("task 1")
        await agent.invoke("task 2")
        assert agent.invoke_count == 2

    @pytest.mark.asyncio
    async def test_invoke_with_thread(self):
        agent = _make_agent("Threaded")
        thread = agent.create_thread()
        result = await agent.invoke("hello", thread=thread)
        assert result["thread_id"] == thread.thread_id
        # Should have user + assistant messages
        assert len(thread.messages) == 2

    @pytest.mark.asyncio
    async def test_invoke_with_context(self):
        agent = _make_agent("Contextual")
        result = await agent.invoke("task", context={"key": "value"})
        assert result["agent"] == "Contextual"

    @pytest.mark.asyncio
    async def test_invoke_elapsed_ms(self):
        agent = _make_agent("Timed")
        result = await agent.invoke("task")
        assert result["elapsed_ms"] >= 0


# ===================================================================
# 6. AgentFrameworkAgent — delegation
# ===================================================================


class TestAgentDelegation:
    @pytest.mark.asyncio
    async def test_delegate_to_connected_agent(self):
        primary = _make_agent("Boss")
        specialist = _make_agent("Worker")
        primary.connect_agent(specialist)
        result = await primary.delegate("Worker", "do this work")
        assert result["agent"] == "Worker"
        assert result["delegated_by"] == "Boss"

    @pytest.mark.asyncio
    async def test_delegate_to_unknown_agent_raises(self):
        primary = _make_agent("Boss")
        with pytest.raises(ValueError, match="not connected"):
            await primary.delegate("Unknown", "task")

    @pytest.mark.asyncio
    async def test_delegate_with_shared_thread(self):
        primary = _make_agent("Boss")
        specialist = _make_agent("Worker")
        primary.connect_agent(specialist)
        thread = primary.create_thread()
        result = await primary.delegate("Worker", "task", thread=thread)
        assert result["thread_id"] == thread.thread_id
        # Thread should have handoff system message + user + assistant
        assert len(thread.messages) >= 2


# ===================================================================
# 7. Sequential Orchestrator
# ===================================================================


class TestSequentialOrchestrator:
    @pytest.mark.asyncio
    async def test_sequential_basic(self):
        agents = [_make_agent("Research"), _make_agent("Build")]
        orch = SequentialOrchestrator(agents)
        result = await orch.run("Build a landing page")
        assert result.pattern == "sequential"
        assert result.status == "completed"
        assert result.agent_count == 2
        assert result.success is True
        assert result.elapsed_ms > 0

    @pytest.mark.asyncio
    async def test_sequential_empty_agents(self):
        orch = SequentialOrchestrator([])
        result = await orch.run("task")
        assert result.status == "failed"
        assert "No agents" in result.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_sequential_single_agent(self):
        orch = SequentialOrchestrator([_make_agent("Solo")])
        result = await orch.run("task")
        assert result.status == "completed"
        assert result.agent_count == 1

    @pytest.mark.asyncio
    async def test_sequential_history(self):
        orch = SequentialOrchestrator([_make_agent("A")])
        await orch.run("task 1")
        await orch.run("task 2")
        assert len(orch.history) == 2

    @pytest.mark.asyncio
    async def test_sequential_passes_output_forward(self):
        agents = [_make_agent("First"), _make_agent("Second")]
        orch = SequentialOrchestrator(agents)
        result = await orch.run("analyze this")
        # Second agent should have received First's output
        assert result.agent_count == 2
        assert result.final_output != ""


# ===================================================================
# 8. Concurrent Orchestrator
# ===================================================================


class TestConcurrentOrchestrator:
    @pytest.mark.asyncio
    async def test_concurrent_basic(self):
        agents = [_make_agent("A"), _make_agent("B"), _make_agent("C")]
        orch = ConcurrentOrchestrator(agents)
        result = await orch.run("research topic X")
        assert result.pattern == "concurrent"
        assert result.status == "completed"
        assert result.agent_count == 3

    @pytest.mark.asyncio
    async def test_concurrent_empty_agents(self):
        orch = ConcurrentOrchestrator([])
        result = await orch.run("task")
        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_concurrent_merges_outputs(self):
        agents = [_make_agent("Alpha"), _make_agent("Beta")]
        orch = ConcurrentOrchestrator(agents)
        result = await orch.run("task")
        assert "[Alpha]" in result.final_output
        assert "[Beta]" in result.final_output

    @pytest.mark.asyncio
    async def test_concurrent_add_remove_agent(self):
        orch = ConcurrentOrchestrator()
        orch.add_agent(_make_agent("X"))
        orch.add_agent(_make_agent("Y"))
        assert len(orch.agents) == 2
        orch.remove_agent("X")
        assert len(orch.agents) == 1


# ===================================================================
# 9. Handoff Orchestrator
# ===================================================================


class TestHandoffOrchestrator:
    @pytest.mark.asyncio
    async def test_handoff_explicit_delegation(self):
        primary = _make_agent("CEO")
        specialist = _make_agent("Builder")
        orch = HandoffOrchestrator(primary=primary, specialists=[specialist])
        result = await orch.run("build a website", delegate_to="Builder")
        assert result.pattern == "handoff"
        assert result.status == "completed"
        assert result.metadata.get("delegated_to") == "Builder"
        assert result.agent_count == 2

    @pytest.mark.asyncio
    async def test_handoff_no_delegation(self):
        primary = _make_agent("CEO")
        orch = HandoffOrchestrator(primary=primary, specialists=[])
        result = await orch.run("simple task")
        assert result.status == "completed"
        assert result.agent_count == 1
        assert result.metadata.get("delegated_to") is None

    @pytest.mark.asyncio
    async def test_handoff_auto_delegation(self):
        primary = _make_agent("CEO")
        # Specialist description matches task keywords
        researcher = AgentFrameworkAgent(
            name="Researcher",
            description="Researches topics and analyzes data from the web",
            instructions="You research things.",
            chat_client=_mock_client(),
        )
        primary.connect_agent(researcher)
        orch = HandoffOrchestrator(primary=primary, specialists=[researcher])
        result = await orch.run("research the latest AI trends and analyze data")
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_handoff_empty_agents(self):
        orch = HandoffOrchestrator()
        result = await orch.run("task")
        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_handoff_primary_property(self):
        p = _make_agent("P")
        s = _make_agent("S")
        orch = HandoffOrchestrator(primary=p, specialists=[s])
        assert orch.primary is p
        assert len(orch.specialists) == 1


# ===================================================================
# 10. MCP Tool Descriptor
# ===================================================================


class TestMCPToolDescriptor:
    @pytest.mark.asyncio
    async def test_execute_with_handler(self):
        tool = MCPToolDescriptor(
            name="test",
            description="test tool",
            parameters={"type": "object"},
            _execute=_echo_handler,
        )
        result = await tool.execute({"key": "val"})
        assert result == {"echoed": {"key": "val"}}

    @pytest.mark.asyncio
    async def test_execute_default_handler(self):
        tool = MCPToolDescriptor(
            name="default",
            description="default tool",
            parameters={"type": "object"},
        )
        result = await tool.execute({"x": 1})
        assert result["status"] == "ok"
        assert result["tool"] == "default"

    def test_to_mcp_dict(self):
        tool = MCPToolDescriptor(
            name="t1",
            description="desc",
            parameters={"type": "object"},
            server="my-server",
            version="2.0.0",
            tags=["tag1", "tag2"],
        )
        d = tool.to_mcp_dict()
        assert d["name"] == "t1"
        assert d["server"] == "my-server"
        assert d["version"] == "2.0.0"
        assert "tag1" in d["tags"]

    def test_to_tool_descriptor(self):
        mcp_tool = MCPToolDescriptor(
            name="mcp_t",
            description="MCP tool",
            parameters={"type": "object"},
            _execute=_echo_handler,
        )
        td = mcp_tool.to_tool_descriptor()
        assert td.name == "mcp_t"
        assert td.execute is not None


# ===================================================================
# 11. MCP Tool Registry
# ===================================================================


class TestMCPToolRegistry:
    def test_register_and_get(self):
        reg = MCPToolRegistry()
        tool = MCPToolDescriptor(name="t1", description="d1", parameters={})
        reg.register(tool)
        assert reg.get("t1") is tool
        assert reg.get("nope") is None

    def test_unregister(self):
        reg = MCPToolRegistry()
        reg.register(MCPToolDescriptor(name="t1", description="d1", parameters={}))
        assert reg.unregister("t1") is True
        assert reg.unregister("t1") is False

    def test_search_by_name(self):
        reg = MCPToolRegistry()
        reg.register(MCPToolDescriptor(name="screenshot", description="take screenshots", parameters={}))
        reg.register(MCPToolDescriptor(name="analysis", description="analyze data", parameters={}))
        results = reg.search("screen")
        assert len(results) == 1
        assert results[0].name == "screenshot"

    def test_search_by_description(self):
        reg = MCPToolRegistry()
        reg.register(MCPToolDescriptor(name="t1", description="convert markdown files", parameters={}))
        results = reg.search("markdown")
        assert len(results) == 1

    def test_search_by_tag(self):
        reg = MCPToolRegistry()
        reg.register(MCPToolDescriptor(name="t1", description="d", parameters={}, tags=["azure", "cloud"]))
        reg.register(MCPToolDescriptor(name="t2", description="d", parameters={}, tags=["local"]))
        results = reg.search_by_tag("azure")
        assert len(results) == 1
        assert results[0].name == "t1"

    def test_list_all(self):
        reg = MCPToolRegistry()
        reg.register(MCPToolDescriptor(name="a", description="d", parameters={}))
        reg.register(MCPToolDescriptor(name="b", description="d", parameters={}))
        assert len(reg.list_all()) == 2

    @pytest.mark.asyncio
    async def test_invoke_success(self):
        reg = MCPToolRegistry()
        reg.register(MCPToolDescriptor(name="echo", description="d", parameters={}, _execute=_echo_handler))
        result = await reg.invoke("echo", {"key": "val"})
        assert result == {"echoed": {"key": "val"}}
        assert len(reg.get_invocation_log()) == 1
        assert reg.get_invocation_log()[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_invoke_not_found(self):
        reg = MCPToolRegistry()
        result = await reg.invoke("nonexistent", {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_invoke_failure(self):
        reg = MCPToolRegistry()
        reg.register(MCPToolDescriptor(name="fail", description="d", parameters={}, _execute=_fail_handler))
        result = await reg.invoke("fail", {})
        assert result["status"] == "error"
        assert len(reg.get_invocation_log()) == 1
        assert reg.get_invocation_log()[0]["status"] == "failed"

    def test_clear(self):
        reg = MCPToolRegistry()
        reg.register(MCPToolDescriptor(name="t1", description="d", parameters={}))
        reg.clear()
        assert len(reg.list_all()) == 0

    def test_to_mcp_list(self):
        reg = MCPToolRegistry()
        reg.register(MCPToolDescriptor(name="t1", description="d1", parameters={}))
        reg.register(MCPToolDescriptor(name="t2", description="d2", parameters={}))
        mcp_list = reg.to_mcp_list()
        assert len(mcp_list) == 2
        assert all("name" in item for item in mcp_list)


# ===================================================================
# 12. Built-in MCP Tools
# ===================================================================


class TestBuiltinMCPTools:
    def test_builtin_tools_count(self):
        assert len(BUILTIN_TOOLS) == 6

    @pytest.mark.asyncio
    async def test_screenshot_tool(self):
        result = await SCREENSHOT_TOOL.execute({"url": "https://example.com"})
        assert result["status"] == "ok"
        assert result["url"] == "https://example.com"
        assert result["format"] == "png"

    @pytest.mark.asyncio
    async def test_ai_analysis_tool(self):
        result = await AI_ANALYSIS_TOOL.execute({"content": "test data", "type": "summary"})
        assert result["status"] == "ok"
        assert result["analysis_type"] == "summary"

    @pytest.mark.asyncio
    async def test_markdown_conversion_tool(self):
        result = await MARKDOWN_CONVERSION_TOOL.execute({"content": "<h1>Hello</h1>", "source_format": "html"})
        assert result["status"] == "ok"
        assert "html" in result["source_format"]

    @pytest.mark.asyncio
    async def test_web_search_tool(self):
        result = await WEB_SEARCH_TOOL.execute({"query": "AI agents", "max_results": 3})
        assert result["status"] == "ok"
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_file_operation_tool(self):
        result = await FILE_OPERATION_TOOL.execute({"operation": "read", "path": "/tmp/test.txt"})
        assert result["status"] == "ok"
        assert result["operation"] == "read"

    @pytest.mark.asyncio
    async def test_api_call_tool(self):
        result = await API_CALL_TOOL.execute({"url": "https://api.example.com/v1", "method": "POST"})
        assert result["status"] == "ok"
        assert result["method"] == "POST"

    def test_create_default_registry(self):
        reg = create_default_registry()
        assert len(reg.list_all()) == 6
        assert reg.get("screenshot") is not None
        assert reg.get("web_search") is not None


# ===================================================================
# 13. A2A Agent Card
# ===================================================================


class TestA2AAgentCard:
    def test_from_agent(self):
        agent = _make_agent("TestBot")
        card = A2AAgentCard.from_agent(agent, "http://localhost:9000")
        assert card.name == "TestBot"
        assert card.url == "http://localhost:9000"
        assert "a2a" in card.protocols
        assert "/a2a" in card.endpoints["jsonrpc"]

    def test_to_dict(self):
        card = A2AAgentCard(name="Bot", description="A bot", url="http://x")
        d = card.to_dict()
        assert d["name"] == "Bot"
        assert d["url"] == "http://x"

    def test_matches_capability_by_name(self):
        card = A2AAgentCard(name="ScreenshotBot", description="Takes screenshots")
        assert card.matches_capability("screenshot") is True
        assert card.matches_capability("analysis") is False

    def test_matches_capability_by_skill(self):
        card = A2AAgentCard(name="Bot", description="General bot", skills=["code", "test"])
        assert card.matches_capability("code") is True
        assert card.matches_capability("design") is False

    def test_matches_capability_by_description(self):
        card = A2AAgentCard(name="Bot", description="Analyzes market trends")
        assert card.matches_capability("market") is True

    def test_agent_card_with_tools(self):
        tool = _make_tool("my_tool")
        agent = _make_agent("ToolBot", tools=[tool])
        card = A2AAgentCard.from_agent(agent)
        assert "my_tool" in card.skills


# ===================================================================
# 14. A2A Server
# ===================================================================


class TestA2AServer:
    def test_server_creation(self):
        agent = _make_agent("ServerAgent")
        server = A2AServer(agent, "http://localhost:8080")
        assert server.agent_card.name == "ServerAgent"

    def test_get_agent_card_dict(self):
        agent = _make_agent("S")
        server = A2AServer(agent)
        card_dict = server.get_agent_card_dict()
        assert card_dict["name"] == "S"
        assert "protocols" in card_dict

    @pytest.mark.asyncio
    async def test_handle_task(self):
        agent = _make_agent("Handler")
        server = A2AServer(agent)
        task = await server.handle_task("do something", from_agent="requester")
        assert task.status == "completed"
        assert task.to_agent == "Handler"
        assert task.from_agent == "requester"
        assert task.result is not None

    @pytest.mark.asyncio
    async def test_handle_task_tracks_tasks(self):
        agent = _make_agent("Tracker")
        server = A2AServer(agent)
        task = await server.handle_task("task 1")
        found = server.get_task(task.task_id)
        assert found is task
        assert len(server.list_tasks()) == 1

    def test_dispatch_agents_info(self):
        agent = _make_agent("InfoAgent")
        server = A2AServer(agent)
        resp = server.dispatch_jsonrpc({
            "jsonrpc": "2.0",
            "method": "agents/info",
            "id": "1",
        })
        assert resp["jsonrpc"] == "2.0"
        assert resp["result"]["name"] == "InfoAgent"
        assert resp["id"] == "1"

    def test_dispatch_tasks_send(self):
        agent = _make_agent("TaskAgent")
        server = A2AServer(agent)
        resp = server.dispatch_jsonrpc({
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {"description": "test task", "from_agent": "client"},
            "id": "2",
        })
        assert resp["result"]["to_agent"] == "TaskAgent"
        assert resp["result"]["status"] == "pending"

    def test_dispatch_tasks_get(self):
        agent = _make_agent("GetAgent")
        server = A2AServer(agent)
        # First create a task
        send_resp = server.dispatch_jsonrpc({
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {"description": "test"},
            "id": "1",
        })
        task_id = send_resp["result"]["task_id"]
        # Then get it
        get_resp = server.dispatch_jsonrpc({
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"task_id": task_id},
            "id": "2",
        })
        assert get_resp["result"]["task_id"] == task_id

    def test_dispatch_invalid_method(self):
        agent = _make_agent("X")
        server = A2AServer(agent)
        resp = server.dispatch_jsonrpc({
            "jsonrpc": "2.0",
            "method": "invalid/method",
            "id": "1",
        })
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_dispatch_invalid_version(self):
        agent = _make_agent("X")
        server = A2AServer(agent)
        resp = server.dispatch_jsonrpc({
            "jsonrpc": "1.0",
            "method": "agents/info",
        })
        assert "error" in resp


# ===================================================================
# 15. A2A Client (unit tests — no real HTTP)
# ===================================================================


class TestA2AClient:
    def test_client_creation(self):
        client = A2AClient(timeout=10.0)
        assert len(client.get_discovered()) == 0

    def test_find_by_capability_empty(self):
        client = A2AClient()
        assert client.find_by_capability("anything") == []


# ===================================================================
# 16. Pre-built Agents
# ===================================================================


class TestPrebuiltAgents:
    def test_create_researcher(self):
        agent = create_researcher_agent(chat_client=_mock_client())
        assert agent.name == "Researcher"
        assert len(agent.list_tools()) == 2
        assert agent.get_tool("web_search") is not None
        assert agent.get_tool("ai_analysis") is not None

    def test_create_analyst(self):
        agent = create_analyst_agent(chat_client=_mock_client())
        assert agent.name == "Analyst"
        assert len(agent.list_tools()) == 2
        assert agent.get_tool("ai_analysis") is not None
        assert agent.get_tool("markdown_conversion") is not None

    def test_create_executor(self):
        agent = create_executor_agent(chat_client=_mock_client())
        assert agent.name == "Executor"
        assert len(agent.list_tools()) == 2
        assert agent.get_tool("file_operation") is not None
        assert agent.get_tool("api_call") is not None

    def test_create_without_tools(self):
        agent = create_researcher_agent(chat_client=_mock_client(), include_tools=False)
        assert len(agent.list_tools()) == 0

    @pytest.mark.asyncio
    async def test_researcher_invoke(self):
        agent = create_researcher_agent(chat_client=_mock_client())
        result = await agent.invoke("Research AI agent frameworks")
        assert result["agent"] == "Researcher"
        assert "response" in result

    @pytest.mark.asyncio
    async def test_analyst_invoke(self):
        agent = create_analyst_agent(chat_client=_mock_client())
        result = await agent.invoke("Analyze the competitive landscape")
        assert result["agent"] == "Analyst"

    @pytest.mark.asyncio
    async def test_executor_invoke(self):
        agent = create_executor_agent(chat_client=_mock_client())
        result = await agent.invoke("Create a configuration file")
        assert result["agent"] == "Executor"


# ===================================================================
# 17. Multi-Agent Orchestration Integration
# ===================================================================


class TestMultiAgentIntegration:
    @pytest.mark.asyncio
    async def test_researcher_analyst_sequential(self):
        """Research → Analyst pipeline."""
        researcher = create_researcher_agent(chat_client=_mock_client())
        analyst = create_analyst_agent(chat_client=_mock_client())
        orch = SequentialOrchestrator([researcher, analyst])
        result = await orch.run("Analyze the AI agent market")
        assert result.status == "completed"
        assert result.agent_count == 2

    @pytest.mark.asyncio
    async def test_all_agents_concurrent(self):
        """All three agents run in parallel."""
        researcher = create_researcher_agent(chat_client=_mock_client())
        analyst = create_analyst_agent(chat_client=_mock_client())
        executor = create_executor_agent(chat_client=_mock_client())
        orch = ConcurrentOrchestrator([researcher, analyst, executor])
        result = await orch.run("Prepare a report on agent infrastructure")
        assert result.status == "completed"
        assert result.agent_count == 3
        assert "[Researcher]" in result.final_output
        assert "[Analyst]" in result.final_output
        assert "[Executor]" in result.final_output

    @pytest.mark.asyncio
    async def test_handoff_researcher_to_executor(self):
        """CEO hands off research results to executor."""
        ceo = _make_agent("CEO")
        researcher = create_researcher_agent(chat_client=_mock_client())
        executor = create_executor_agent(chat_client=_mock_client())
        orch = HandoffOrchestrator(primary=ceo, specialists=[researcher, executor])
        result = await orch.run("Research AI tools then create a summary file", delegate_to="Executor")
        assert result.status == "completed"
        assert result.metadata["delegated_to"] == "Executor"

    @pytest.mark.asyncio
    async def test_three_stage_pipeline(self):
        """Research → Analyze → Execute pipeline."""
        researcher = create_researcher_agent(chat_client=_mock_client())
        analyst = create_analyst_agent(chat_client=_mock_client())
        executor = create_executor_agent(chat_client=_mock_client())
        orch = SequentialOrchestrator([researcher, analyst, executor])
        result = await orch.run("Build a competitive analysis document")
        assert result.status == "completed"
        assert result.agent_count == 3

    @pytest.mark.asyncio
    async def test_a2a_server_with_prebuilt_agent(self):
        """A2A server wrapping a pre-built agent."""
        researcher = create_researcher_agent(chat_client=_mock_client())
        server = A2AServer(researcher, "http://localhost:9000")
        card = server.get_agent_card_dict()
        assert card["name"] == "Researcher"
        assert "web_search" in card["skills"]

        task = await server.handle_task("Research quantum computing")
        assert task.status == "completed"

    @pytest.mark.asyncio
    async def test_orchestrator_result_properties(self):
        """Test OrchestratorResult dataclass properties."""
        result = OrchestratorResult(pattern="test", task="test task", status="completed")
        assert result.success is True
        assert result.agent_count == 0

        result.status = "failed"
        assert result.success is False


# ===================================================================
# 18. Edge Cases and Error Handling
# ===================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_agent_message_metadata(self):
        msg = AgentMessage(role="user", content="test", metadata={"key": "val"})
        assert msg.metadata["key"] == "val"
        assert msg.timestamp > 0

    def test_a2a_task_defaults(self):
        task = A2ATask(description="test")
        assert task.status == "pending"
        assert task.result is None
        assert task.task_id.startswith("a2a_")

    @pytest.mark.asyncio
    async def test_registry_invoke_tracking(self):
        """Test that invocations are tracked in order."""
        reg = MCPToolRegistry()
        reg.register(MCPToolDescriptor(name="t1", description="d", parameters={}, _execute=_echo_handler))
        reg.register(MCPToolDescriptor(name="t2", description="d", parameters={}, _execute=_echo_handler))
        await reg.invoke("t1", {"a": 1})
        await reg.invoke("t2", {"b": 2})
        await reg.invoke("t1", {"c": 3})
        log = reg.get_invocation_log()
        assert len(log) == 3
        assert log[0]["tool"] == "t1"
        assert log[1]["tool"] == "t2"
        assert log[2]["tool"] == "t1"

    def test_thread_metadata(self):
        agent = _make_agent("M")
        thread = agent.create_thread(purpose="testing", priority="high")
        assert thread.metadata["purpose"] == "testing"
        assert thread.metadata["priority"] == "high"

    @pytest.mark.asyncio
    async def test_concurrent_preserves_all_results(self):
        """Ensure concurrent orchestration keeps all agent results."""
        agents = [_make_agent(f"Agent{i}") for i in range(5)]
        orch = ConcurrentOrchestrator(agents)
        result = await orch.run("test task")
        assert result.agent_count == 5
        agent_names = [r["agent"] for r in result.agent_results]
        for i in range(5):
            assert f"Agent{i}" in agent_names
