"""Analyst Agent — Analyzes data and content using AI reasoning.

Pre-built agent for the Microsoft Agent Framework integration.
Specializes in data analysis, comparison, and structured reasoning.
"""

from __future__ import annotations

from typing import Any

from src.framework.agent import AgentFrameworkAgent, ToolDescriptor
from src.framework.mcp_tools import AI_ANALYSIS_TOOL, MARKDOWN_CONVERSION_TOOL
from src.config import get_chat_client

ANALYST_INSTRUCTIONS = """You are an Analyst Agent in the AgentOS multi-agent system.

Your role is to analyze data, content, and information to produce actionable insights.

Capabilities:
1. **Data Analysis**: Analyze structured and unstructured data
2. **Comparison**: Compare options, approaches, or solutions
3. **Reasoning**: Apply structured reasoning to complex problems
4. **Report Formatting**: Convert analysis results to clean Markdown

When given an analysis task:
1. Identify the type of analysis needed (comparison, summary, evaluation, etc.)
2. Break down the data or content into analyzable components
3. Apply appropriate analysis frameworks
4. Present findings as:
   - Summary (2-3 sentences)
   - Key insights (bulleted)
   - Recommendation with confidence score
   - Trade-offs or risks identified

Be precise, data-driven, and transparent about assumptions.
"""


def create_analyst_agent(
    chat_client: Any = None,
    include_tools: bool = True,
) -> AgentFrameworkAgent:
    """Create an Analyst agent with AI analysis and formatting tools.

    Args:
        chat_client: Optional chat client. Uses mock if None.
        include_tools: Whether to include default tools (default True).

    Returns:
        Configured AgentFrameworkAgent for analysis tasks.
    """
    tools = []
    if include_tools:
        tools = [
            AI_ANALYSIS_TOOL.to_tool_descriptor(),
            MARKDOWN_CONVERSION_TOOL.to_tool_descriptor(),
        ]

    return AgentFrameworkAgent(
        name="Analyst",
        description="Analyzes data and content using AI reasoning, produces structured insights and comparisons",
        instructions=ANALYST_INSTRUCTIONS,
        tools=tools,
        chat_client=chat_client or get_chat_client(),
        model="default",
    )
