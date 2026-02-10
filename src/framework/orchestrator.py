"""Orchestration patterns for Microsoft Agent Framework integration.

Implements three core orchestration modes that map to Microsoft's patterns:
- Sequential: Pipe a task through agents in order
- Concurrent: Run multiple agents on the same task, merge results
- Handoff: Agent A dynamically delegates to Agent B

Each orchestrator works with AgentFrameworkAgent instances and tracks
execution state, timing, and results.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.framework.agent import AgentFrameworkAgent, AgentThread


@dataclass
class OrchestratorResult:
    """Aggregated result from an orchestration run."""

    orchestration_id: str = field(default_factory=lambda: f"orch_{uuid.uuid4().hex[:12]}")
    pattern: str = ""  # "sequential", "concurrent", "handoff"
    task: str = ""
    status: str = "pending"  # "pending", "running", "completed", "failed"
    agent_results: list[dict[str, Any]] = field(default_factory=list)
    final_output: str = ""
    elapsed_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "completed"

    @property
    def agent_count(self) -> int:
        return len(self.agent_results)


class Orchestrator:
    """Base orchestrator that coordinates multiple AgentFrameworkAgent instances."""

    def __init__(self, agents: list[AgentFrameworkAgent] | None = None) -> None:
        self._agents: list[AgentFrameworkAgent] = agents or []
        self._runs: list[OrchestratorResult] = []

    def add_agent(self, agent: AgentFrameworkAgent) -> None:
        self._agents.append(agent)

    def remove_agent(self, name: str) -> bool:
        before = len(self._agents)
        self._agents = [a for a in self._agents if a.name != name]
        return len(self._agents) < before

    @property
    def agents(self) -> list[AgentFrameworkAgent]:
        return list(self._agents)

    @property
    def history(self) -> list[OrchestratorResult]:
        return list(self._runs)

    async def run(self, task: str, **kwargs: Any) -> OrchestratorResult:
        raise NotImplementedError("Subclasses must implement run()")


class SequentialOrchestrator(Orchestrator):
    """Sequential orchestration: pipe task through agents in order.

    Each agent receives the previous agent's output as additional context.
    Maps to Microsoft Agent Framework's sequential pipeline pattern.
    """

    async def run(
        self,
        task: str,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> OrchestratorResult:
        result = OrchestratorResult(pattern="sequential", task=task)
        result.status = "running"
        t0 = time.time()

        if not self._agents:
            result.status = "failed"
            result.metadata["error"] = "No agents configured"
            result.elapsed_ms = (time.time() - t0) * 1000
            self._runs.append(result)
            return result

        current_input = task
        shared_thread = thread or AgentThread()

        try:
            for agent in self._agents:
                agent_result = await agent.invoke(
                    current_input,
                    thread=shared_thread,
                    context={"orchestration": "sequential", "step": len(result.agent_results) + 1},
                )
                result.agent_results.append(agent_result)

                # Pass this agent's response as input to the next
                current_input = (
                    f"Previous agent ({agent.name}) output:\n"
                    f"{agent_result['response']}\n\n"
                    f"Original task: {task}"
                )

            result.status = "completed"
            # Final output is the last agent's response
            if result.agent_results:
                result.final_output = result.agent_results[-1].get("response", "")
        except Exception as exc:
            result.status = "failed"
            result.metadata["error"] = str(exc)

        result.elapsed_ms = round((time.time() - t0) * 1000, 2)
        self._runs.append(result)
        return result


class ConcurrentOrchestrator(Orchestrator):
    """Concurrent orchestration: run all agents on the same task in parallel.

    All agents process the task independently and results are aggregated.
    Maps to Microsoft Agent Framework's concurrent/fan-out pattern.
    """

    async def run(
        self,
        task: str,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> OrchestratorResult:
        result = OrchestratorResult(pattern="concurrent", task=task)
        result.status = "running"
        t0 = time.time()

        if not self._agents:
            result.status = "failed"
            result.metadata["error"] = "No agents configured"
            result.elapsed_ms = (time.time() - t0) * 1000
            self._runs.append(result)
            return result

        try:
            # Run all agents concurrently
            tasks = [
                agent.invoke(
                    task,
                    context={"orchestration": "concurrent", "agent_index": i},
                )
                for i, agent in enumerate(self._agents)
            ]
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, ar in enumerate(agent_results):
                if isinstance(ar, Exception):
                    result.agent_results.append({
                        "agent": self._agents[i].name,
                        "status": "failed",
                        "error": str(ar),
                    })
                else:
                    result.agent_results.append(ar)

            # Merge outputs from all agents
            outputs = []
            for ar in result.agent_results:
                if isinstance(ar, dict) and "response" in ar:
                    outputs.append(f"[{ar.get('agent', 'unknown')}]: {ar['response']}")
            result.final_output = "\n\n".join(outputs)

            has_failures = any(
                isinstance(ar, dict) and ar.get("status") == "failed"
                for ar in result.agent_results
            )
            result.status = "completed" if not has_failures else "partial"

        except Exception as exc:
            result.status = "failed"
            result.metadata["error"] = str(exc)

        result.elapsed_ms = round((time.time() - t0) * 1000, 2)
        self._runs.append(result)
        return result


class HandoffOrchestrator(Orchestrator):
    """Handoff orchestration: primary agent delegates to specialists dynamically.

    The first agent in the list is the primary (orchestrator). It can
    delegate tasks to any other agent in the list via the Connected Agents
    pattern. Maps to Microsoft Agent Framework's handoff/delegation mode.

    The primary agent decides what to delegate based on keyword matching
    against connected agent descriptions (using mock LLM).
    """

    def __init__(
        self,
        primary: AgentFrameworkAgent | None = None,
        specialists: list[AgentFrameworkAgent] | None = None,
    ) -> None:
        all_agents = []
        if primary:
            all_agents.append(primary)
        if specialists:
            all_agents.extend(specialists)
        super().__init__(all_agents)

        # Wire up connected agents
        if primary and specialists:
            for s in specialists:
                primary.connect_agent(s)

    @property
    def primary(self) -> AgentFrameworkAgent | None:
        return self._agents[0] if self._agents else None

    @property
    def specialists(self) -> list[AgentFrameworkAgent]:
        return self._agents[1:] if len(self._agents) > 1 else []

    async def run(
        self,
        task: str,
        thread: AgentThread | None = None,
        delegate_to: str | None = None,
        **kwargs: Any,
    ) -> OrchestratorResult:
        """Run the handoff orchestration.

        Args:
            task: The task to process.
            thread: Optional shared thread.
            delegate_to: If specified, the primary agent will delegate
                        to this specialist. If None, the primary processes
                        the task itself first, then determines delegation.
        """
        result = OrchestratorResult(pattern="handoff", task=task)
        result.status = "running"
        t0 = time.time()

        if not self._agents:
            result.status = "failed"
            result.metadata["error"] = "No agents configured"
            result.elapsed_ms = (time.time() - t0) * 1000
            self._runs.append(result)
            return result

        primary = self._agents[0]
        shared_thread = thread or primary.create_thread()

        try:
            # Step 1: Primary agent processes the task
            primary_result = await primary.invoke(task, thread=shared_thread)
            result.agent_results.append(primary_result)

            # Step 2: Determine delegation target
            target_name = delegate_to
            if target_name is None:
                target_name = self._select_delegate(task)

            # Step 3: Delegate if a target was identified
            if target_name and primary.get_connected_agent(target_name):
                delegate_result = await primary.delegate(
                    target_name, task, thread=shared_thread,
                )
                result.agent_results.append(delegate_result)
                result.final_output = delegate_result.get("response", "")
                result.metadata["delegated_to"] = target_name
            else:
                result.final_output = primary_result.get("response", "")
                result.metadata["delegated_to"] = None

            result.status = "completed"

        except Exception as exc:
            result.status = "failed"
            result.metadata["error"] = str(exc)

        result.elapsed_ms = round((time.time() - t0) * 1000, 2)
        self._runs.append(result)
        return result

    def _select_delegate(self, task: str) -> str | None:
        """Select the best specialist for a task using keyword matching.

        In production, this would use the LLM to make the decision.
        For the mock implementation, we match task keywords against
        specialist descriptions.
        """
        task_lower = task.lower()
        best_match: str | None = None
        best_score = 0

        for specialist in self.specialists:
            score = 0
            desc_words = specialist.description.lower().split()
            for word in desc_words:
                if len(word) > 3 and word in task_lower:
                    score += 1
            if score > best_score:
                best_score = score
                best_match = specialist.name

        return best_match
