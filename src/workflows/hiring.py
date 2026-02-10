"""Agent Hiring workflow: Discover → Evaluate → Hire → Execute → Pay.

CEO discovers external agents from the marketplace, evaluates their
capabilities against task requirements, hires one, sends the task via
A2A protocol, receives the result, and records payment.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Any

import httpx

from src.mcp_servers.registry_server import registry, AgentCard
from src.mcp_servers.payment_hub import ledger
from src.metrics.collector import get_metrics_collector
from src.llm import get_llm_client


@dataclass
class HiringDecision:
    """Records why a particular agent was selected (or rejected)."""
    agent_name: str
    approved: bool
    reason: str
    price_usdc: float
    capability_match: float  # 0.0-1.0 relevance score


@dataclass
class HiringResult:
    """Full result of a hiring workflow execution."""
    task_id: str
    task_description: str
    status: str  # "completed" | "failed" | "no_agents_found"
    discovery: list[dict[str, Any]]
    decision: dict[str, Any] | None
    external_result: dict[str, Any] | None
    payment: dict[str, Any] | None
    budget_summary: dict[str, Any]
    elapsed_s: float


def discover_external_agents(capability: str, max_price: float | None = None) -> list[AgentCard]:
    """Search the registry for hireable external agents."""
    all_matches = registry.search(capability, max_price)
    return [a for a in all_matches if a.is_external]


def evaluate_agent(agent: AgentCard, required_skills: list[str]) -> HiringDecision:
    """Score an external agent against required skills.

    Uses LLM-powered matching when Azure OpenAI is available, falls back
    to deterministic skill overlap otherwise.
    """
    agent_skills_lower = {s.lower() for s in agent.skills}
    required_lower = {s.lower() for s in required_skills}

    if not required_lower:
        match_score = 1.0
    else:
        overlap = agent_skills_lower & required_lower
        match_score = len(overlap) / len(required_lower)

    price = float(agent.price_per_call.replace("$", ""))

    # Use LLM for richer evaluation when available
    llm = get_llm_client()
    if llm.is_azure and required_skills:
        try:
            llm_result = llm.job_match(
                candidate_profile={"skills": agent.skills, "name": agent.name},
                job_requirements={"required_skills": required_skills},
            )
            llm_score = llm_result.get("match_score", match_score)
            reasoning = llm_result.get("reasoning", "")
            if isinstance(llm_score, (int, float)) and 0.0 <= llm_score <= 1.0:
                match_score = llm_score
            if reasoning:
                return HiringDecision(
                    agent_name=agent.name,
                    approved=match_score >= 0.3,
                    reason=f"AI evaluation: {reasoning}",
                    price_usdc=price,
                    capability_match=match_score,
                )
        except Exception:
            pass  # Fall through to rule-based

    return HiringDecision(
        agent_name=agent.name,
        approved=match_score >= 0.3,
        reason=f"Skill match {match_score:.0%} — matched {overlap or 'general capability'}",
        price_usdc=price,
        capability_match=match_score,
    )


async def send_task_to_agent(
    agent: AgentCard,
    task_id: str,
    description: str,
    budget: float,
) -> dict[str, Any]:
    """Send a task to an external agent via its A2A endpoint.

    Uses httpx to POST to the agent's ``/a2a/tasks`` endpoint.
    """
    payload = {
        "task_id": task_id,
        "description": description,
        "from_agent": "ceo",
        "budget": budget,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{agent.endpoint}/a2a/tasks", json=payload)
        resp.raise_for_status()
        return resp.json()


async def run_hiring_workflow(
    task_id: str,
    task_description: str,
    required_skills: list[str],
    budget_usd: float,
    capability_query: str | None = None,
    max_price: float | None = None,
) -> HiringResult:
    """Execute the full agent hiring workflow.

    Steps:
    1. Allocate budget
    2. Discover external agents matching capability
    3. Evaluate each candidate
    4. Select best agent
    5. Send task via A2A
    6. Record payment
    7. Return aggregated result
    """
    t0 = time.monotonic()
    query = capability_query or (required_skills[0] if required_skills else "general")

    # 1. Budget allocation
    ledger.allocate_budget(task_id, budget_usd)

    # 2. Discovery
    candidates = discover_external_agents(query, max_price)
    discovery_data = [asdict(c) for c in candidates]

    if not candidates:
        return HiringResult(
            task_id=task_id,
            task_description=task_description,
            status="no_agents_found",
            discovery=discovery_data,
            decision=None,
            external_result=None,
            payment=None,
            budget_summary=_budget_summary(task_id),
            elapsed_s=round(time.monotonic() - t0, 3),
        )

    # 3. Evaluate candidates
    decisions: list[HiringDecision] = []
    for agent in candidates:
        decision = evaluate_agent(agent, required_skills)
        decisions.append(decision)

    # 4. Select best agent (highest capability match among approved)
    approved = [d for d in decisions if d.approved]
    if not approved:
        return HiringResult(
            task_id=task_id,
            task_description=task_description,
            status="no_suitable_agents",
            discovery=discovery_data,
            decision=asdict(decisions[0]) if decisions else None,
            external_result=None,
            payment=None,
            budget_summary=_budget_summary(task_id),
            elapsed_s=round(time.monotonic() - t0, 3),
        )

    best = max(approved, key=lambda d: d.capability_match)
    selected_agent = registry.get(best.agent_name)

    # Check budget before hiring
    if best.price_usdc > ledger.get_budget(task_id).remaining:
        return HiringResult(
            task_id=task_id,
            task_description=task_description,
            status="insufficient_budget",
            discovery=discovery_data,
            decision=asdict(best),
            external_result=None,
            payment=None,
            budget_summary=_budget_summary(task_id),
            elapsed_s=round(time.monotonic() - t0, 3),
        )

    # 5. Send task to external agent
    try:
        external_result = await send_task_to_agent(
            selected_agent, task_id, task_description, budget_usd,
        )
    except Exception as exc:
        return HiringResult(
            task_id=task_id,
            task_description=task_description,
            status="failed",
            discovery=discovery_data,
            decision=asdict(best),
            external_result={"error": str(exc)},
            payment=None,
            budget_summary=_budget_summary(task_id),
            elapsed_s=round(time.monotonic() - t0, 3),
        )

    # 6. Record payment
    payment_record = ledger.record_payment(
        from_agent="ceo",
        to_agent=best.agent_name,
        amount=best.price_usdc,
        task_id=task_id,
    )

    # 7. Record metrics
    elapsed = round(time.monotonic() - t0, 3)
    mc = get_metrics_collector()
    mc.update_metrics({
        "task_id": task_id,
        "agent_id": best.agent_name,
        "task_type": "hiring",
        "status": "success",
        "cost_usdc": best.price_usdc,
        "latency_ms": elapsed * 1000,
    })
    mc.record_payment({
        "to_agent": best.agent_name,
        "task_id": task_id,
        "amount_usdc": payment_record.amount_usdc,
        "status": payment_record.status,
    })

    # 8. Aggregate
    return HiringResult(
        task_id=task_id,
        task_description=task_description,
        status="completed",
        discovery=discovery_data,
        decision=asdict(best),
        external_result=external_result,
        payment={
            "tx_id": payment_record.tx_id,
            "amount_usdc": payment_record.amount_usdc,
            "to_agent": payment_record.to_agent,
            "status": payment_record.status,
        },
        budget_summary=_budget_summary(task_id),
        elapsed_s=elapsed,
    )


def _budget_summary(task_id: str) -> dict[str, Any]:
    """Return a budget summary dict for the given task."""
    budget = ledger.get_budget(task_id)
    if budget is None:
        return {"allocated": 0.0, "spent": 0.0, "remaining": 0.0}
    return {
        "allocated": budget.allocated,
        "spent": budget.spent,
        "remaining": budget.remaining,
    }
