"""Executor Agent — Takes action by creating files, calling APIs, etc.

Pre-built agent for the Microsoft Agent Framework integration.
Specializes in executing tasks: file operations, API calls, and deployments.
"""

from __future__ import annotations

from typing import Any

from src.framework.agent import AgentFrameworkAgent, ToolDescriptor
from src.framework.mcp_tools import FILE_OPERATION_TOOL, API_CALL_TOOL
from src.config import get_chat_client

EXECUTOR_INSTRUCTIONS = """You are an Executor Agent in the AgentOS multi-agent system.

Your role is to take concrete actions based on plans and instructions from other agents.

Capabilities:
1. **File Operations**: Create, read, update, and delete files
2. **API Calls**: Make HTTP requests to external services
3. **Task Execution**: Execute multi-step action plans

When given an execution task:
1. Validate the action plan and required inputs
2. Execute each step in the correct order
3. Handle errors gracefully with retries where appropriate
4. Report results with:
   - Actions taken (list of steps completed)
   - Files created or modified
   - API responses received
   - Any errors encountered

Be precise, safe, and report exactly what was done.
Never execute destructive actions without explicit confirmation.
"""


def create_executor_agent(
    chat_client: Any = None,
    include_tools: bool = True,
) -> AgentFrameworkAgent:
    """Create an Executor agent with file and API tools.

    Args:
        chat_client: Optional chat client. Uses mock if None.
        include_tools: Whether to include default tools (default True).

    Returns:
        Configured AgentFrameworkAgent for execution tasks.
    """
    tools = []
    if include_tools:
        tools = [
            FILE_OPERATION_TOOL.to_tool_descriptor(),
            API_CALL_TOOL.to_tool_descriptor(),
        ]

    return AgentFrameworkAgent(
        name="Executor",
        description="Executes actions including file operations, API calls, and multi-step task execution",
        instructions=EXECUTOR_INSTRUCTIONS,
        tools=tools,
        chat_client=chat_client or get_chat_client(),
        model="default",
    )
