"""Researcher Agent — Searches the web and synthesizes information.

Pre-built agent for the Microsoft Agent Framework integration.
Uses web search and AI analysis MCP tools to research topics
and produce structured reports.
"""

from __future__ import annotations

from typing import Any

from src.framework.agent import AgentFrameworkAgent, ToolDescriptor
from src.framework.mcp_tools import WEB_SEARCH_TOOL, AI_ANALYSIS_TOOL
from src.config import get_chat_client

RESEARCHER_INSTRUCTIONS = """You are a Researcher Agent in the AgentOS multi-agent system.

Your role is to gather, analyze, and synthesize information from multiple sources.

Capabilities:
1. **Web Search**: Find relevant information, documentation, and data
2. **AI Analysis**: Analyze gathered content for insights and patterns
3. **Report Generation**: Produce structured reports with findings

When given a research task:
1. Break the topic into searchable queries
2. Search for relevant information
3. Analyze and cross-reference findings
4. Produce a structured report with:
   - Key findings (3-5 bullet points)
   - Sources (with URLs)
   - Confidence level (high/medium/low)
   - Recommendations

Always cite sources and indicate confidence levels.
"""


def create_researcher_agent(
    chat_client: Any = None,
    include_tools: bool = True,
) -> AgentFrameworkAgent:
    """Create a Researcher agent with web search and analysis tools.

    Args:
        chat_client: Optional chat client. Uses mock if None.
        include_tools: Whether to include default tools (default True).

    Returns:
        Configured AgentFrameworkAgent for research tasks.
    """
    tools = []
    if include_tools:
        tools = [
            WEB_SEARCH_TOOL.to_tool_descriptor(),
            AI_ANALYSIS_TOOL.to_tool_descriptor(),
        ]

    return AgentFrameworkAgent(
        name="Researcher",
        description="Researches topics using web search and AI analysis, produces structured reports",
        instructions=RESEARCHER_INSTRUCTIONS,
        tools=tools,
        chat_client=chat_client or get_chat_client(),
        model="default",
    )
