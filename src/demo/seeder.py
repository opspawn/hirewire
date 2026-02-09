"""Seed realistic demo data into the AgentOS database.

Populates tasks, payments, and agents so judges see a rich dashboard
on first load rather than an empty screen.
"""

from __future__ import annotations

import random
import time
import uuid

from src.mcp_servers.payment_hub import ledger
from src.mcp_servers.registry_server import registry, AgentCard
from src.storage import get_storage

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
# Completed demo tasks
# ---------------------------------------------------------------------------

COMPLETED_TASKS: list[dict] = [
    {"description": "Build landing page for AI startup", "budget": 3.50, "workflow": "ceo"},
    {"description": "Analyze competitor pricing across 5 SaaS tools", "budget": 2.00, "workflow": "ceo"},
    {"description": "Design logo and brand identity kit", "budget": 1.50, "workflow": "ceo"},
    {"description": "Write API documentation for payment endpoints", "budget": 1.00, "workflow": "ceo"},
    {"description": "Deploy microservice to production with CI/CD", "budget": 5.00, "workflow": "ceo"},
]

# ---------------------------------------------------------------------------
# Active demo tasks (pending/running)
# ---------------------------------------------------------------------------

ACTIVE_TASKS: list[dict] = [
    {"description": "Research best vector databases for agent memory", "budget": 1.50, "status": "running"},
    {"description": "Build real-time WebSocket dashboard for agent metrics", "budget": 4.00, "status": "running"},
    {"description": "Evaluate agent marketplace pricing strategies", "budget": 0.75, "status": "pending"},
]

# ---------------------------------------------------------------------------
# Agent names used for payments
# ---------------------------------------------------------------------------

AGENT_NAMES = ["ceo", "builder", "research", "designer-ext-001", "designer-ext-002", "analyst-ext-001"]


def seed_demo_data() -> dict:
    """Populate the database with realistic demo data.

    Returns a summary dict with counts of seeded items.
    """
    storage = get_storage()
    now = time.time()
    tasks_created = 0
    payments_created = 0
    agents_registered = 0

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

    # 2. Create completed tasks (spread over the last hour)
    for i, t in enumerate(COMPLETED_TASKS):
        task_id = f"demo_{uuid.uuid4().hex[:8]}"
        created = now - (3600 - i * 600)  # spaced ~10 min apart over last hour
        storage.save_task(
            task_id=task_id,
            description=t["description"],
            workflow=t["workflow"],
            budget_usd=t["budget"],
            status="completed",
            created_at=created,
            result={
                "original_task": t["description"],
                "subtasks": [
                    {"id": "s1", "description": f"Phase 1: {t['description']}", "agent": "research"},
                    {"id": "s2", "description": f"Phase 2: Execute {t['description']}", "agent": "builder"},
                ],
                "execution_order": "sequential",
                "estimated_cost": round(t["budget"] * 0.6, 2),
                "complexity": "moderate",
                "status": "planned",
            },
        )
        tasks_created += 1

        # Create 1-2 payment transactions per completed task
        cost = round(t["budget"] * random.uniform(0.3, 0.7), 4)
        ledger.allocate_budget(task_id, t["budget"])

        # Payment from CEO to worker
        to_agent = random.choice(["builder", "research", "designer-ext-001", "analyst-ext-001"])
        ledger.record_payment(from_agent="ceo", to_agent=to_agent, amount=cost, task_id=task_id)
        payments_created += 1

        # Sometimes a second payment for multi-agent tasks
        if random.random() > 0.5:
            cost2 = round(cost * random.uniform(0.2, 0.5), 4)
            to2 = random.choice([a for a in AGENT_NAMES if a not in ("ceo", to_agent)])
            ledger.record_payment(from_agent="ceo", to_agent=to2, amount=cost2, task_id=task_id)
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

        # Running tasks may have a partial payment
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

    return {
        "tasks_created": tasks_created,
        "payments_created": payments_created,
        "agents_registered": agents_registered,
    }
