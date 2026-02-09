"""Live demo runner — submits tasks at intervals for real-time dashboard activity.

When started, the DemoRunner submits a curated task every 30 seconds,
triggering the full CEO -> analyze -> hire -> pay pipeline so judges
can watch the dashboard update in real time.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from src.agents.ceo_agent import analyze_task
from src.mcp_servers.payment_hub import ledger
from src.storage import get_storage

# Curated list of interesting demo tasks
DEMO_TASK_LIST: list[dict[str, Any]] = [
    {"description": "Build a React dashboard for real-time agent monitoring", "budget": 4.00},
    {"description": "Research top 10 AI agent frameworks and compare features", "budget": 2.00},
    {"description": "Create a Stripe integration for agent marketplace payments", "budget": 3.50},
    {"description": "Design a mobile-first onboarding flow for new users", "budget": 1.50},
    {"description": "Deploy a Redis cache layer for agent state management", "budget": 2.50},
    {"description": "Analyze customer churn patterns using transaction data", "budget": 1.75},
    {"description": "Build an automated testing pipeline for MCP tool servers", "budget": 3.00},
    {"description": "Write a technical blog post about agent-to-agent payments", "budget": 1.00},
    {"description": "Implement OAuth2 authentication for the agent marketplace", "budget": 4.50},
    {"description": "Create a performance benchmarking suite for agent workflows", "budget": 2.25},
]


class DemoRunner:
    """Submits curated tasks at regular intervals for live demo activity."""

    def __init__(self, interval: float = 30.0) -> None:
        self.interval = interval
        self._task: asyncio.Task | None = None
        self._running = False
        self._task_index = 0
        self._tasks_submitted = 0

    @property
    def is_running(self) -> bool:
        return self._running

    async def run(self) -> None:
        """Main demo loop — submit a task, wait, repeat."""
        self._running = True
        try:
            while self._running:
                await self._submit_next_task()
                self._task_index = (self._task_index + 1) % len(DEMO_TASK_LIST)
                self._tasks_submitted += 1
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False

    async def _submit_next_task(self) -> None:
        """Submit the next demo task through the full pipeline."""
        spec = DEMO_TASK_LIST[self._task_index]
        task_id = f"demo_{uuid.uuid4().hex[:8]}"
        storage = get_storage()
        now = time.time()

        # 1. Create the task
        storage.save_task(
            task_id=task_id,
            description=spec["description"],
            workflow="ceo",
            budget_usd=spec["budget"],
            status="pending",
            created_at=now,
        )

        # 2. CEO analyzes the task
        storage.update_task_status(task_id, "running")
        analysis = await analyze_task(spec["description"])

        # 3. Allocate budget and record payment
        estimated_cost = analysis.get("estimated_cost", 0.0)
        if estimated_cost > 0:
            ledger.allocate_budget(task_id, spec["budget"])
            ledger.record_payment(
                from_agent="ceo",
                to_agent="builder",
                amount=min(estimated_cost, spec["budget"]),
                task_id=task_id,
            )

        # 4. Mark completed
        storage.update_task_status(task_id, "completed", result=analysis)

    def start(self) -> None:
        """Start the demo loop as a background asyncio task."""
        if self._running or (self._task is not None and not self._task.done()):
            return  # already running
        self._running = True
        self._task = asyncio.create_task(self.run())

    def stop(self) -> None:
        """Stop the demo loop."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    def status(self) -> dict[str, Any]:
        """Return current demo runner status."""
        return {
            "running": self._running,
            "tasks_submitted": self._tasks_submitted,
            "next_task_index": self._task_index,
            "interval_seconds": self.interval,
        }
