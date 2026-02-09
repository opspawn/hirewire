"""CEO Agent - The orchestrator that analyzes tasks, manages budget, and delegates work.

Uses ChatAgent with tools for task analysis, budget checking, and agent hiring.
"""

from __future__ import annotations

import re
from typing import Any

from agent_framework import ChatAgent, tool

from src.config import get_chat_client, get_settings
from src.mcp_servers.payment_hub import ledger

CEO_INSTRUCTIONS = """You are the CEO Agent of AgentOS, an autonomous agent operating system.

Your responsibilities:
1. **Task Analysis**: When a user submits a task, break it down into subtasks
   and determine which specialist agents are needed.
2. **Hiring Decisions**: Decide whether to use internal agents (Builder, Research)
   or hire external agents from the marketplace via x402 micropayments.
3. **Budget Management**: Track spending against budget constraints. Never exceed
   the allocated budget for a task.
4. **Quality Control**: Review results from sub-agents and determine if the task
   is complete or needs revision.
5. **Delegation**: Route subtasks to the appropriate agents with clear instructions.

Available internal agents:
- Builder Agent: Writes code, runs tests, deploys services
- Research Agent: Searches the web, analyzes data, produces reports

Decision framework:
- If the task is purely research, delegate to Research Agent
- If the task requires code, delegate to Builder Agent
- For complex tasks, use sequential (research first, then build) or parallel execution
- If an internal agent can't handle it, search the marketplace for an external agent
- Always check budget before hiring external agents

When you receive a task, respond with a structured plan:
1. Task breakdown (subtasks)
2. Agent assignments
3. Execution order (sequential/parallel)
4. Estimated cost (if external agents needed)
5. Success criteria
"""


# --- CEO Tools ---

@tool(name="analyze_task", description="Break down a task into subtasks with agent assignments")
async def analyze_task(task_description: str) -> dict[str, Any]:
    """Analyze a task and return a structured breakdown.

    Parses the task description to detect task types using keyword matching
    and returns appropriate subtask breakdowns with cost estimates.
    """
    try:
        desc_lower = task_description.lower()
        words = desc_lower.split()
        word_count = len(words)

        # Keyword sets for detecting task types
        research_keywords = {
            "search", "find", "compare", "analyze", "research", "investigate",
            "evaluate", "review", "assess", "study", "explore", "look",
            "report", "summarize", "survey", "benchmark", "discover",
        }
        build_keywords = {
            "build", "create", "implement", "deploy", "code", "write",
            "develop", "fix", "refactor", "test", "install", "setup",
            "configure", "migrate", "update", "upgrade", "ship", "launch",
        }

        has_research = any(kw in desc_lower for kw in research_keywords)
        has_build = any(kw in desc_lower for kw in build_keywords)

        # Determine task type and build subtasks
        subtasks: list[dict[str, Any]] = []

        if has_research and has_build:
            # Complex task: research first, then build
            subtasks = [
                {
                    "id": "research",
                    "description": f"Research phase: gather information for '{task_description}'",
                    "agent": "research",
                },
                {
                    "id": "build",
                    "description": f"Build phase: implement based on research for '{task_description}'",
                    "agent": "builder",
                },
            ]
            execution_order = "sequential"
        elif has_research:
            subtasks = [
                {
                    "id": "research",
                    "description": f"Research: {task_description}",
                    "agent": "research",
                },
            ]
            execution_order = "parallel"
        elif has_build:
            subtasks = [
                {
                    "id": "build",
                    "description": f"Build: {task_description}",
                    "agent": "builder",
                },
            ]
            execution_order = "parallel"
        else:
            # Default: treat as research + build sequential
            subtasks = [
                {
                    "id": "research",
                    "description": f"Research: {task_description}",
                    "agent": "research",
                },
                {
                    "id": "build",
                    "description": f"Build: {task_description}",
                    "agent": "builder",
                },
            ]
            execution_order = "sequential"

        # Estimate cost based on complexity (word count)
        if word_count <= 10:
            estimated_cost = 0.10
            complexity = "simple"
        elif word_count <= 30:
            estimated_cost = 0.25
            complexity = "moderate"
        elif word_count <= 60:
            estimated_cost = 0.50
            complexity = "complex"
        else:
            estimated_cost = 1.00
            complexity = "very_complex"

        # Scale cost by number of subtasks
        estimated_cost *= len(subtasks)

        return {
            "original_task": task_description,
            "subtasks": subtasks,
            "execution_order": execution_order,
            "estimated_cost": round(estimated_cost, 2),
            "complexity": complexity,
            "task_type": "research+build" if (has_research and has_build) else
                         "research" if has_research else
                         "build" if has_build else "general",
            "status": "planned",
        }
    except Exception as e:
        return {
            "original_task": task_description,
            "subtasks": [],
            "execution_order": "sequential",
            "estimated_cost": 0.0,
            "status": "error",
            "error": str(e),
        }


@tool(name="check_budget", description="Check remaining budget for the current task")
async def check_budget(task_id: str) -> dict[str, Any]:
    """Check budget allocation for a task using the real PaymentLedger."""
    try:
        budget = ledger.get_budget(task_id)
        if budget is None:
            return {
                "task_id": task_id,
                "allocated": 0.0,
                "spent": 0.0,
                "remaining": 0.0,
                "currency": "USDC",
                "message": f"No budget allocated for task '{task_id}'. Call allocate_budget first.",
            }
        return {
            "task_id": task_id,
            "allocated": budget.allocated,
            "spent": budget.spent,
            "remaining": budget.remaining,
            "currency": "USDC",
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "error": f"Failed to check budget: {e}",
        }


@tool(name="approve_hire", description="Approve hiring an external agent for a subtask")
async def approve_hire(
    agent_name: str,
    subtask_id: str,
    price: float,
    max_budget: float = 1.0,
) -> dict[str, Any]:
    """Approve or reject hiring an external agent based on budget."""
    if price > max_budget:
        return {
            "approved": False,
            "reason": f"Price ${price:.2f} exceeds budget ${max_budget:.2f}",
        }
    return {
        "approved": True,
        "agent": agent_name,
        "subtask_id": subtask_id,
        "price": price,
    }


def create_ceo_agent(chat_client=None) -> ChatAgent:
    """Create and return the CEO agent.

    Args:
        chat_client: Optional ChatClientProtocol instance. If None, creates one
                     from environment config.
    """
    if chat_client is None:
        chat_client = get_chat_client()

    return ChatAgent(
        chat_client=chat_client,
        name="CEO",
        description="Orchestrator agent that analyzes tasks, manages budget, and delegates work",
        instructions=CEO_INSTRUCTIONS,
        tools=[analyze_task, check_budget, approve_hire],
    )
