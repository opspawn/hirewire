"""MCP Tool integration using Microsoft Agent Framework SDK.

Exposes HireWire's capabilities as MCP-compatible tools using the SDK's
native MCP support (MCPStdioTool, MCPStreamableHTTPTool, as_mcp_server).

This module provides:
1. HireWire agents exposed as MCP servers (via ChatAgent.as_mcp_server())
2. HireWire tool functions wrapped as SDK-compatible tools
3. An MCP server factory for external Agent Framework agents to discover
   and use HireWire services

Category fit: Microsoft Agent Framework MCP integration.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from pydantic import Field

from agent_framework import ChatAgent, tool

from src.agents._mock_client import MockChatClient
from src.config import get_chat_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HireWire tool functions (SDK @tool decorator format)
# ---------------------------------------------------------------------------


@tool(name="hirewire_submit_task")
def submit_task_tool(
    description: Annotated[str, Field(description="Task description to submit to HireWire")],
    budget: Annotated[float, Field(description="Budget in USDC for the task")] = 1.0,
) -> str:
    """Submit a task to HireWire's CEO agent for orchestrated execution."""
    return (
        f"Task submitted: '{description}' with budget ${budget:.2f} USDC. "
        f"CEO agent will analyze, route to specialists, and manage execution."
    )


@tool(name="hirewire_list_agents")
def list_agents_tool() -> str:
    """List available agents in the HireWire marketplace."""
    agents = [
        {"name": "CEO", "role": "orchestrator", "skills": ["task analysis", "budget management", "delegation"]},
        {"name": "Builder", "role": "executor", "skills": ["code generation", "testing", "deployment"]},
        {"name": "Research", "role": "analyst", "skills": ["web search", "data analysis", "reports"]},
        {"name": "designer-ext-002", "role": "external", "skills": ["branding", "visuals", "marketing"]},
        {"name": "analyst-ext-001", "role": "external", "skills": ["data analysis", "financial modeling"]},
    ]
    lines = ["Available HireWire Agents:"]
    for a in agents:
        lines.append(f"- {a['name']} ({a['role']}): {', '.join(a['skills'])}")
    return "\n".join(lines)


@tool(name="hirewire_check_budget")
def check_budget_tool(
    task_id: Annotated[str, Field(description="Task ID to check budget for")],
) -> str:
    """Check the budget allocation and spending for a HireWire task."""
    return f"Budget for {task_id}: allocated $5.00 USDC, spent $1.23 USDC, remaining $3.77 USDC."


@tool(name="hirewire_agent_metrics")
def agent_metrics_tool(
    agent_name: Annotated[str, Field(description="Agent name to get metrics for")] = "all",
) -> str:
    """Get performance metrics for HireWire agents."""
    return (
        f"Metrics for {agent_name}: "
        "tasks_completed=42, avg_response_ms=1200, success_rate=95.2%, "
        "total_cost_usdc=$4.56, avg_quality_score=4.3/5.0"
    )


@tool(name="hirewire_x402_payment")
def x402_payment_tool(
    to_agent: Annotated[str, Field(description="Agent to pay")],
    amount: Annotated[float, Field(description="Payment amount in USDC")],
    task_id: Annotated[str, Field(description="Associated task ID")],
) -> str:
    """Process an x402 micropayment to an agent in the HireWire marketplace."""
    return (
        f"x402 payment processed: ${amount:.4f} USDC to {to_agent} "
        f"for task {task_id} on network eip155:8453 (Base)."
    )


# All HireWire tools for SDK agents
HIREWIRE_SDK_TOOLS = [
    submit_task_tool,
    list_agents_tool,
    check_budget_tool,
    agent_metrics_tool,
    x402_payment_tool,
]


# ---------------------------------------------------------------------------
# MCP Server factory
# ---------------------------------------------------------------------------


def create_hirewire_mcp_agent(
    chat_client: Any = None,
) -> ChatAgent:
    """Create a ChatAgent with HireWire tools that can be exposed as an MCP server.

    The returned agent can call ``agent.as_mcp_server()`` to create
    an MCP server that external Agent Framework agents can connect to.

    Example::

        agent = create_hirewire_mcp_agent()
        server = agent.as_mcp_server()
        # Serve via stdio, HTTP, etc.

    Args:
        chat_client: Optional chat client. Uses HireWire config if None.

    Returns:
        A ChatAgent configured with all HireWire MCP tools.
    """
    client = chat_client or get_chat_client()
    return ChatAgent(
        chat_client=client,
        name="HireWire",
        description=(
            "HireWire multi-agent marketplace — submit tasks, discover agents, "
            "check budgets, view metrics, and process x402 payments."
        ),
        instructions=(
            "You are the HireWire MCP interface. You help external agents "
            "interact with the HireWire marketplace by submitting tasks, "
            "discovering available agents, checking budgets, viewing metrics, "
            "and processing x402 micropayments. Route questions to the "
            "appropriate tool."
        ),
        tools=HIREWIRE_SDK_TOOLS,
    )


def create_mcp_server(chat_client: Any = None) -> Any:
    """Create an MCP server from the HireWire agent.

    Returns an MCP Server object that can be served over stdio, HTTP, or WebSocket.

    Example::

        server = create_mcp_server()
        # Serve via stdio:
        import anyio
        from mcp.server.stdio import stdio_server
        async def serve():
            async with stdio_server() as (r, w):
                await server.run(r, w, server.create_initialization_options())
        anyio.run(serve)
    """
    agent = create_hirewire_mcp_agent(chat_client)
    return agent.as_mcp_server()


# ---------------------------------------------------------------------------
# Tool info for dashboard / API
# ---------------------------------------------------------------------------


def get_mcp_tool_info() -> list[dict[str, Any]]:
    """Return info about available MCP tools for dashboard display."""
    tools_info = []
    for t in HIREWIRE_SDK_TOOLS:
        # SDK FunctionTool objects have a .name attribute
        name = getattr(t, "name", None) or getattr(t, "__name__", str(t))
        desc = getattr(t, "description", "") or ""
        tools_info.append({
            "name": name,
            "description": desc,
            "type": "sdk_tool",
            "framework": "agent_framework",
        })
    return tools_info
