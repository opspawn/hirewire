"""Research Agent - Searches the web, analyzes information, and produces reports.

Uses ChatAgent with tools for web search and data analysis.
Supports handoff pattern for receiving work from CEO agent.
"""

from __future__ import annotations

from typing import Any

from agent_framework import ChatAgent, tool

from src.config import get_chat_client

RESEARCH_INSTRUCTIONS = """You are the Research Agent of AgentOS.

Your responsibilities:
1. **Web Search**: Find relevant information, documentation, and resources.
2. **Analysis**: Analyze data, compare options, and synthesize findings.
3. **Reports**: Produce structured reports with key findings and recommendations.
4. **Market Research**: Evaluate agent marketplace offerings, pricing, and capabilities.

When you receive a research task:
1. Identify what information is needed
2. Search for relevant sources
3. Analyze and cross-reference findings
4. Produce a structured report with:
   - Key findings
   - Sources cited
   - Recommendations
   - Confidence level (high/medium/low)

You specialize in:
- Technology research (frameworks, APIs, tools)
- Market analysis (competitors, pricing, trends)
- Agent capability assessment
- Documentation summarization
"""


@tool(name="web_search", description="Search the web for information")
async def web_search(
    query: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search the web for information using DuckDuckGo.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default: 5).
    """
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", r.get("link", "")),
                "snippet": r.get("body", r.get("snippet", "")),
            }
            for r in raw_results
        ]

        return {
            "query": query,
            "results": results,
            "total_results": len(results),
        }
    except ImportError:
        return {
            "query": query,
            "results": [],
            "total_results": 0,
            "error": "ddgs package not installed. Run: pip install ddgs",
        }
    except Exception as e:
        return {
            "query": query,
            "results": [],
            "total_results": 0,
            "error": f"Search failed: {e}",
        }


@tool(name="analyze_data", description="Analyze structured data and produce insights")
async def analyze_data(
    data_description: str,
    analysis_type: str = "summary",
) -> dict[str, Any]:
    """Analyze data and return insights.

    Placeholder - will integrate with real analysis capabilities.
    """
    return {
        "analysis_type": analysis_type,
        "input": data_description,
        "findings": [f"Analysis of: {data_description}"],
        "confidence": "medium",
    }


@tool(name="search_marketplace", description="Search the agent marketplace for capable agents")
async def search_marketplace(
    capability: str,
    max_price: float = 1.0,
) -> dict[str, Any]:
    """Search the agent marketplace for agents with specific capabilities.

    Placeholder - will integrate with real marketplace registry.
    """
    return {
        "capability": capability,
        "max_price": max_price,
        "agents_found": [],
        "message": "Marketplace search placeholder - no external agents registered yet",
    }


def create_research_agent(chat_client=None) -> ChatAgent:
    """Create and return the Research agent.

    Args:
        chat_client: Optional ChatClientProtocol instance. If None, creates one
                     from environment config.
    """
    if chat_client is None:
        chat_client = get_chat_client()

    return ChatAgent(
        chat_client=chat_client,
        name="Research",
        description="Research agent that searches the web, analyzes data, and produces reports",
        instructions=RESEARCH_INSTRUCTIONS,
        tools=[web_search, analyze_data, search_marketplace],
    )
