"""Hiring Flow — discover, select, pay, assign, verify, release.

Orchestrates the full lifecycle of hiring an agent from the marketplace:
1. discover() — find agents with required skills
2. select() — pick the best candidate
3. negotiate_price() — agree on price within budget
4. pay() — escrow payment via x402
5. assign() — send task to the agent
6. verify_result() — check the result
7. release_payment() — release escrow on success

Integrates with:
- MarketplaceRegistry for agent discovery
- SkillMatcher for ranking
- AgentEscrow for payment lifecycle
- Orchestrator for multi-agent task routing
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

from src.marketplace import MarketplaceRegistry, SkillMatcher, AgentListing, marketplace
from src.marketplace.x402 import AgentEscrow, EscrowEntry, X402PaymentGate, PaymentConfig


@dataclass
class HireRequest:
    """A request to hire an agent for a task."""

    task_id: str = field(default_factory=lambda: f"hire_{uuid.uuid4().hex[:12]}")
    description: str = ""
    required_skills: list[str] = field(default_factory=list)
    budget: float = 0.0  # Max USDC to spend
    requester: str = "ceo"


@dataclass
class HireResult:
    """Result of a hiring flow."""

    task_id: str = ""
    status: str = "pending"  # pending, hired, completed, failed, budget_exceeded, no_agents
    agent_id: str = ""
    agent_name: str = ""
    escrow_id: str = ""
    agreed_price: float = 0.0
    task_result: dict[str, Any] | None = None
    error: str = ""
    elapsed_s: float = 0.0
    budget_remaining: float = 0.0


class BudgetTracker:
    """Tracks spending per requester across tasks."""

    def __init__(self, total_budget: float = 100.0) -> None:
        self._total_budget = total_budget
        self._spent: dict[str, float] = {}  # task_id -> amount

    @property
    def total_budget(self) -> float:
        return self._total_budget

    @property
    def total_spent(self) -> float:
        return sum(self._spent.values())

    @property
    def remaining(self) -> float:
        return self._total_budget - self.total_spent

    def can_afford(self, amount: float) -> bool:
        """Check if the budget can cover this amount."""
        return amount <= self.remaining

    def spend(self, task_id: str, amount: float) -> bool:
        """Record spending. Returns False if budget exceeded."""
        if not self.can_afford(amount):
            return False
        self._spent[task_id] = self._spent.get(task_id, 0.0) + amount
        return True

    def get_spending(self, task_id: str) -> float:
        """Get total spending for a task."""
        return self._spent.get(task_id, 0.0)

    def spending_report(self) -> dict[str, Any]:
        """Generate a spending report."""
        return {
            "total_budget": self._total_budget,
            "total_spent": self.total_spent,
            "remaining": self.remaining,
            "tasks": dict(self._spent),
        }

    def reset(self) -> None:
        """Reset all spending."""
        self._spent.clear()


class HiringManager:
    """Orchestrates the full hiring lifecycle with payment integration."""

    def __init__(
        self,
        registry: MarketplaceRegistry | None = None,
        escrow: AgentEscrow | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> None:
        self._registry = registry or marketplace
        self._matcher = SkillMatcher(self._registry)
        self._escrow = escrow or AgentEscrow()
        self._budget = budget_tracker or BudgetTracker()
        self._hires: list[HireResult] = []

    @property
    def escrow(self) -> AgentEscrow:
        return self._escrow

    @property
    def budget(self) -> BudgetTracker:
        return self._budget

    @property
    def hire_history(self) -> list[HireResult]:
        return list(self._hires)

    def discover(self, request: HireRequest) -> list[tuple[AgentListing, float]]:
        """Step 1: Discover agents matching the request."""
        return self._matcher.match(
            required_skills=request.required_skills,
            max_price=request.budget if request.budget > 0 else None,
        )

    def select(self, candidates: list[tuple[AgentListing, float]]) -> AgentListing | None:
        """Step 2: Select the best candidate (highest score)."""
        if not candidates:
            return None
        return candidates[0][0]

    def negotiate_price(self, agent: AgentListing, max_budget: float) -> float | None:
        """Step 3: Determine the agreed price. Returns None if too expensive."""
        if agent.price_per_unit > max_budget:
            return None
        return agent.price_per_unit

    def pay(self, request: HireRequest, agent: AgentListing, price: float) -> EscrowEntry | None:
        """Step 4: Create escrow payment. Returns None if budget exceeded."""
        if not self._budget.can_afford(price):
            return None
        self._budget.spend(request.task_id, price)
        entry = self._escrow.hold_payment(
            payer=request.requester,
            payee=agent.name,
            amount=price,
            task_id=request.task_id,
        )
        return entry

    def assign(self, agent: AgentListing, request: HireRequest) -> dict[str, Any]:
        """Step 5: Assign task to agent (mock — returns simulated result)."""
        return {
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "task_id": request.task_id,
            "description": request.description,
            "status": "completed",
            "output": f"Task '{request.description}' completed by {agent.name}",
        }

    def verify_result(self, result: dict[str, Any]) -> bool:
        """Step 6: Verify the result is acceptable."""
        return result.get("status") == "completed"

    def release_payment(self, escrow_id: str) -> EscrowEntry | None:
        """Step 7: Release escrowed payment on successful completion."""
        return self._escrow.release_on_completion(escrow_id)

    def refund_payment(self, escrow_id: str) -> EscrowEntry | None:
        """Refund escrowed payment on failure."""
        return self._escrow.refund_on_failure(escrow_id)

    def hire(self, request: HireRequest) -> HireResult:
        """Execute the full hiring flow end-to-end.

        Runs: discover -> select -> negotiate -> pay -> assign -> verify -> release
        """
        t0 = time.time()
        result = HireResult(task_id=request.task_id)

        # 1. Discover
        candidates = self.discover(request)
        if not candidates:
            result.status = "no_agents"
            result.error = "No agents found matching required skills"
            result.elapsed_s = round(time.time() - t0, 4)
            result.budget_remaining = self._budget.remaining
            self._hires.append(result)
            return result

        # 2. Select
        agent = self.select(candidates)
        result.agent_id = agent.agent_id
        result.agent_name = agent.name

        # 3. Negotiate price
        price = self.negotiate_price(agent, request.budget)
        if price is None:
            result.status = "budget_exceeded"
            result.error = f"Agent price ${agent.price_per_unit} exceeds budget ${request.budget}"
            result.elapsed_s = round(time.time() - t0, 4)
            result.budget_remaining = self._budget.remaining
            self._hires.append(result)
            return result

        result.agreed_price = price

        # 4. Pay (escrow)
        escrow_entry = self.pay(request, agent, price)
        if escrow_entry is None:
            result.status = "budget_exceeded"
            result.error = "Global budget exhausted"
            result.elapsed_s = round(time.time() - t0, 4)
            result.budget_remaining = self._budget.remaining
            self._hires.append(result)
            return result

        result.escrow_id = escrow_entry.escrow_id
        result.status = "hired"

        # 5. Assign task
        task_result = self.assign(agent, request)
        result.task_result = task_result

        # 6. Verify
        if self.verify_result(task_result):
            # 7. Release payment
            self.release_payment(escrow_entry.escrow_id)
            result.status = "completed"
            # Update marketplace stats
            self._registry.increment_jobs(agent.agent_id)
        else:
            # Refund on failure
            self.refund_payment(escrow_entry.escrow_id)
            result.status = "failed"
            result.error = "Task result verification failed"

        result.elapsed_s = round(time.time() - t0, 4)
        result.budget_remaining = self._budget.remaining
        self._hires.append(result)
        return result


__all__ = [
    "HireRequest",
    "HireResult",
    "BudgetTracker",
    "HiringManager",
]
