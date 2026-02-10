"""Seed realistic demo data into the HireWire database.

Populates tasks, payments, and agents so judges see a rich dashboard
on first load rather than an empty screen.

When HIREWIRE_DEMO=1, the startup hook calls seed_demo_data() which:
1. Registers external demo agents
2. Creates completed tasks with REAL GPT-4o responses (when Azure is available)
3. Records x402 payment transactions for each task
4. Creates a few active/pending tasks for visual variety
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from typing import Any

from src.mcp_servers.payment_hub import ledger
from src.mcp_servers.registry_server import registry, AgentCard
from src.storage import get_storage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demo agents (external mock agents beyond the 3 built-in)
# ---------------------------------------------------------------------------

DEMO_AGENTS: list[dict] = [
    {
        "name": "designer-ext-002",
        "description": "Creates brand identities, marketing visuals, and presentation decks",
        "skills": ["design", "branding", "presentations", "marketing"],
        "price_per_call": "$0.03",
        "endpoint": "http://127.0.0.1:9200",
        "protocol": "a2a",
        "payment": "x402",
        "is_external": True,
        "metadata": {"provider": "BrandCraft AI", "rating": 4.6, "tasks_completed": 87},
    },
    {
        "name": "analyst-ext-001",
        "description": "Performs data analysis, competitive research, and financial modeling",
        "skills": ["analysis", "data", "finance", "market-research", "reports"],
        "price_per_call": "$0.04",
        "endpoint": "http://127.0.0.1:9300",
        "protocol": "a2a",
        "payment": "x402",
        "is_external": True,
        "metadata": {"provider": "DataMind AI", "rating": 4.9, "tasks_completed": 215},
    },
]

# ---------------------------------------------------------------------------
# Demo scenarios — realistic business tasks with agent routing
# ---------------------------------------------------------------------------

DEMO_SCENARIOS: list[dict[str, Any]] = [
    {
        "description": "Analyze competitor pricing across top 5 AI agent platforms",
        "budget": 2.50,
        "agents": ["research", "analyst-ext-001"],
        "task_type": "research",
        "orchestration_pattern": "sdk_sequential",
        "sdk_pattern_detail": "SequentialBuilder: Research → Analyst pipeline via Agent Framework SDK",
        "gpt_prompt": (
            "You are a market research analyst. Analyze competitor pricing for AI agent platforms. "
            "Compare pricing models of Fixie, CrewAI, AutoGen, LangGraph, and SuperAGI. "
            "Return a concise competitive analysis with pricing tiers and key differentiators. "
            "Keep response under 200 words."
        ),
    },
    {
        "description": "Design landing page mockup for HireWire agent marketplace",
        "budget": 3.00,
        "agents": ["designer-ext-001", "builder"],
        "task_type": "build",
        "orchestration_pattern": "sdk_handoff",
        "sdk_pattern_detail": "HandoffBuilder: CEO delegates to Designer, then hands off to Builder via Agent Framework SDK",
        "gpt_prompt": (
            "You are a UI/UX designer. Describe a landing page design for HireWire, an AI agent marketplace "
            "where agents can hire other agents using x402 micropayments. Include: hero section copy, "
            "3 key feature sections, CTA button text, and color scheme recommendations. "
            "Keep response under 200 words."
        ),
    },
    {
        "description": "Research market trends in autonomous AI agent infrastructure",
        "budget": 1.75,
        "agents": ["research"],
        "task_type": "research",
        "orchestration_pattern": "native",
        "gpt_prompt": (
            "You are a technology analyst. Summarize the current market trends in AI agent infrastructure "
            "for Q1 2026. Cover: agent-to-agent protocols, micropayment adoption, MCP tool ecosystem, "
            "and enterprise agent orchestration. Include 3 key insights. "
            "Keep response under 200 words."
        ),
    },
    {
        "description": "Build automated testing pipeline for x402 payment verification",
        "budget": 4.00,
        "agents": ["builder", "research"],
        "task_type": "research+build",
        "orchestration_pattern": "sdk_sequential",
        "sdk_pattern_detail": "SequentialBuilder: Research → Builder pipeline via Agent Framework SDK",
        "gpt_prompt": (
            "You are a senior software engineer. Outline a testing pipeline for x402 micropayment verification "
            "in an agent marketplace. Cover: unit tests for escrow logic, integration tests for payment flow, "
            "mock facilitator setup, and CI/CD integration steps. "
            "Keep response under 200 words."
        ),
    },
    {
        "description": "Evaluate agent scoring algorithms for marketplace optimization",
        "budget": 2.00,
        "agents": ["analyst-ext-001", "research"],
        "task_type": "research",
        "orchestration_pattern": "native",
        "gpt_prompt": (
            "You are a data scientist. Compare Thompson Sampling vs UCB1 vs Epsilon-Greedy for "
            "optimizing agent hiring in a marketplace. Consider: convergence speed, exploration/exploitation "
            "tradeoff, cold start handling, and computational cost. Recommend the best approach. "
            "Keep response under 200 words."
        ),
    },
]

# Active tasks (pending/running, no GPT-4o needed)
ACTIVE_TASKS: list[dict] = [
    {"description": "Research best vector databases for agent memory", "budget": 1.50, "status": "running"},
    {"description": "Build real-time WebSocket dashboard for agent metrics", "budget": 4.00, "status": "running"},
    {"description": "Evaluate agent marketplace pricing strategies", "budget": 0.75, "status": "pending"},
]

AGENT_NAMES = ["ceo", "builder", "research", "designer-ext-001", "designer-ext-002", "analyst-ext-001"]


def _get_gpt4o_response(prompt: str) -> str | None:
    """Call Azure OpenAI GPT-4o for a real response. Returns None if unavailable."""
    try:
        from src.framework.azure_llm import azure_available, get_azure_llm
        if not azure_available():
            return None
        provider = get_azure_llm()
        result = provider.chat_completion(
            messages=[
                {"role": "system", "content": "You are a HireWire AI agent providing professional analysis. Be concise and actionable."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return result.get("content", "")
    except Exception as e:
        logger.warning("GPT-4o call failed during demo seed: %s", e)
        return None


def seed_demo_data() -> dict:
    """Populate the database with realistic demo data.

    If Azure OpenAI (GPT-4o) is available, tasks will contain real AI-generated
    responses. Otherwise, falls back to structured mock results.

    Returns a summary dict with counts of seeded items.
    """
    storage = get_storage()
    now = time.time()
    tasks_created = 0
    payments_created = 0
    agents_registered = 0
    gpt4o_responses = 0

    # 1. Register additional demo agents
    for agent_def in DEMO_AGENTS:
        if registry.get(agent_def["name"]) is None:
            registry.register(AgentCard(
                name=agent_def["name"],
                description=agent_def["description"],
                skills=agent_def["skills"],
                price_per_call=agent_def["price_per_call"],
                endpoint=agent_def["endpoint"],
                protocol=agent_def["protocol"],
                payment=agent_def["payment"],
                is_external=agent_def["is_external"],
                metadata=agent_def["metadata"],
            ))
            agents_registered += 1

    # 2. Create completed tasks with real GPT-4o responses
    for i, scenario in enumerate(DEMO_SCENARIOS):
        task_id = f"demo_{uuid.uuid4().hex[:8]}"
        created = now - (3600 - i * 600)  # spaced ~10 min apart over last hour

        # Get real GPT-4o response
        gpt_response = _get_gpt4o_response(scenario["gpt_prompt"])
        if gpt_response:
            gpt4o_responses += 1

        # Build rich result
        primary_agent = scenario["agents"][0]
        secondary_agent = scenario["agents"][1] if len(scenario["agents"]) > 1 else None
        subtasks = [
            {
                "id": "s1",
                "description": f"Phase 1: Research & analysis for '{scenario['description']}'",
                "agent": primary_agent,
                "status": "completed",
            },
        ]
        if secondary_agent:
            subtasks.append({
                "id": "s2",
                "description": f"Phase 2: Execute & deliver for '{scenario['description']}'",
                "agent": secondary_agent,
                "status": "completed",
            })

        estimated_cost = round(scenario["budget"] * random.uniform(0.3, 0.7), 4)

        orch_pattern = scenario.get("orchestration_pattern", "native")
        result = {
            "original_task": scenario["description"],
            "subtasks": subtasks,
            "execution_order": "sequential" if secondary_agent else "parallel",
            "orchestration_pattern": orch_pattern,
            "sdk_pattern_detail": scenario.get("sdk_pattern_detail", ""),
            "estimated_cost": estimated_cost,
            "complexity": "moderate" if scenario["budget"] < 3.0 else "complex",
            "task_type": scenario["task_type"],
            "status": "completed",
            "agent_response": gpt_response or f"Analysis complete for: {scenario['description']}. Key findings documented.",
            "agent_response_preview": (gpt_response or "Analysis complete.")[:150],
            "model": "gpt-4o" if gpt_response else "mock",
            "response_time_ms": round(random.uniform(800, 3500), 0),
        }

        storage.save_task(
            task_id=task_id,
            description=scenario["description"],
            workflow="ceo",
            budget_usd=scenario["budget"],
            status="completed",
            created_at=created,
            result=result,
        )
        tasks_created += 1

        # Create payment transactions
        ledger.allocate_budget(task_id, scenario["budget"])

        # Primary agent payment
        ledger.record_payment(
            from_agent="ceo",
            to_agent=primary_agent,
            amount=estimated_cost,
            task_id=task_id,
        )
        payments_created += 1

        # Secondary agent payment (x402 external)
        if secondary_agent and secondary_agent.startswith(("designer", "analyst")):
            ext_cost = round(estimated_cost * random.uniform(0.2, 0.5), 4)
            ledger.record_payment(
                from_agent="ceo",
                to_agent=secondary_agent,
                amount=ext_cost,
                task_id=task_id,
            )
            payments_created += 1

    # 3. Create active tasks
    for j, t in enumerate(ACTIVE_TASKS):
        task_id = f"demo_{uuid.uuid4().hex[:8]}"
        created = now - (120 + j * 30)  # recent
        storage.save_task(
            task_id=task_id,
            description=t["description"],
            workflow="ceo",
            budget_usd=t["budget"],
            status=t["status"],
            created_at=created,
        )
        tasks_created += 1

        if t["status"] == "running":
            partial = round(t["budget"] * 0.2, 4)
            ledger.allocate_budget(task_id, t["budget"])
            ledger.record_payment(
                from_agent="ceo",
                to_agent="builder",
                amount=partial,
                task_id=task_id,
            )
            payments_created += 1

    # Count SDK-orchestrated tasks
    sdk_tasks = sum(
        1 for s in DEMO_SCENARIOS
        if s.get("orchestration_pattern", "").startswith("sdk_")
    )

    logger.info(
        "Demo seeded: %d tasks (%d SDK-orchestrated), %d payments, %d agents, %d GPT-4o responses",
        tasks_created, sdk_tasks, payments_created, agents_registered, gpt4o_responses,
    )

    return {
        "tasks_created": tasks_created,
        "payments_created": payments_created,
        "agents_registered": agents_registered,
        "gpt4o_responses": gpt4o_responses,
        "sdk_orchestrated_tasks": sdk_tasks,
    }
