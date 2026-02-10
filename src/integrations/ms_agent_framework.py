"""Microsoft Agent Framework SDK orchestration integration for HireWire.

Uses the REAL Agent Framework SDK (agent_framework v1.0.0b260130) to
orchestrate HireWire's agents through native SDK patterns:

- SequentialBuilder  — pipeline orchestration (Research → Build → Deploy)
- ConcurrentBuilder  — parallel fan-out (multiple analysts)
- HandoffBuilder     — dynamic delegation between agents

This module bridges HireWire's existing AgentFrameworkAgent abstraction
with the SDK's native workflow builders, adding real SDK orchestration
as an option alongside the existing custom orchestrator.

Category fit: Microsoft Agent Framework SDK orchestration patterns.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from agent_framework import (
    ChatAgent,
    ChatMessage,
    HandoffBuilder,
    Role,
    SequentialBuilder,
    ConcurrentBuilder,
    WorkflowOutputEvent,
)

from src.agents._mock_client import MockChatClient
from src.config import get_chat_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass — unified result format
# ---------------------------------------------------------------------------


@dataclass
class SDKOrchestrationResult:
    """Result from an SDK-native orchestration run."""

    orchestration_id: str = field(
        default_factory=lambda: f"sdk_orch_{uuid.uuid4().hex[:12]}"
    )
    pattern: str = ""  # "sdk_sequential", "sdk_concurrent", "sdk_handoff"
    task: str = ""
    status: str = "pending"
    agent_results: list[dict[str, Any]] = field(default_factory=list)
    final_output: str = ""
    elapsed_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    sdk_version: str = "agent_framework==1.0.0b260130"

    @property
    def success(self) -> bool:
        return self.status == "completed"


# ---------------------------------------------------------------------------
# Agent factory — create SDK ChatAgent instances
# ---------------------------------------------------------------------------


def create_sdk_agent(
    name: str,
    instructions: str,
    description: str = "",
    chat_client: Any = None,
    tools: list | None = None,
) -> ChatAgent:
    """Create a native Agent Framework SDK ChatAgent.

    Args:
        name: Agent name (e.g. 'CEO', 'Builder', 'Research').
        instructions: System instructions for the agent.
        description: Short description for handoff routing.
        chat_client: Chat client backend. Uses HireWire config if None.
        tools: Optional list of tool functions.

    Returns:
        A configured ``agent_framework.ChatAgent`` instance.
    """
    client = chat_client or get_chat_client()
    return ChatAgent(
        chat_client=client,
        name=name,
        description=description or f"HireWire {name} agent",
        instructions=instructions,
        tools=tools or [],
    )


# ---------------------------------------------------------------------------
# Pre-built SDK agents for HireWire
# ---------------------------------------------------------------------------


def get_hirewire_sdk_agents(
    chat_client: Any = None,
) -> dict[str, ChatAgent]:
    """Create the standard HireWire agent roster as SDK ChatAgents.

    Returns a dict keyed by role name with configured ChatAgent instances.
    """
    client = chat_client or get_chat_client()

    ceo = create_sdk_agent(
        name="CEO",
        description="Orchestrator that analyzes tasks, manages budget, and delegates work.",
        instructions=(
            "You are the CEO Agent in HireWire. Your role is to:\n"
            "1. Analyze incoming tasks and break them into subtasks\n"
            "2. Estimate costs and allocate budget\n"
            "3. Route work to the best agent (Builder, Research, external specialists)\n"
            "4. Review results and ensure quality\n"
            "Be concise and structured in your analysis."
        ),
        chat_client=client,
    )

    builder = create_sdk_agent(
        name="Builder",
        description="Code generation, testing, and deployment specialist.",
        instructions=(
            "You are the Builder Agent in HireWire. Your role is to:\n"
            "1. Write clean, tested code based on requirements\n"
            "2. Run tests and fix issues\n"
            "3. Deploy services and manage infrastructure\n"
            "Respond with structured implementation plans and deliverables."
        ),
        chat_client=client,
    )

    research = create_sdk_agent(
        name="Research",
        description="Web search, data analysis, and competitive research specialist.",
        instructions=(
            "You are the Research Agent in HireWire. Your role is to:\n"
            "1. Search for relevant information and data\n"
            "2. Analyze findings and identify patterns\n"
            "3. Produce structured reports with key insights\n"
            "Always cite sources and indicate confidence levels."
        ),
        chat_client=client,
    )

    analyst = create_sdk_agent(
        name="Analyst",
        description="Financial modeling, pricing analysis, and market research.",
        instructions=(
            "You are the Analyst Agent in HireWire. Your role is to:\n"
            "1. Perform quantitative analysis on market data\n"
            "2. Build pricing models and competitive comparisons\n"
            "3. Generate data-driven recommendations\n"
            "Be precise with numbers and transparent about assumptions."
        ),
        chat_client=client,
    )

    return {
        "ceo": ceo,
        "builder": builder,
        "research": research,
        "analyst": analyst,
    }


# ---------------------------------------------------------------------------
# SDK Orchestration: Sequential
# ---------------------------------------------------------------------------


async def run_sequential(
    agents: list[ChatAgent],
    task: str,
    **kwargs: Any,
) -> SDKOrchestrationResult:
    """Run a task through agents sequentially using the SDK SequentialBuilder.

    Each agent receives the conversation history from the previous agent,
    building on their output. Maps to Microsoft's sequential pipeline pattern.

    Args:
        agents: Ordered list of ChatAgents to pipeline the task through.
        task: The task description to process.

    Returns:
        SDKOrchestrationResult with per-agent outputs and final result.
    """
    result = SDKOrchestrationResult(pattern="sdk_sequential", task=task)
    result.status = "running"
    t0 = time.time()

    if not agents:
        result.status = "failed"
        result.metadata["error"] = "No agents provided"
        result.elapsed_ms = round((time.time() - t0) * 1000, 2)
        return result

    try:
        workflow = SequentialBuilder().participants(agents).build()

        messages: list[dict[str, Any]] = []
        async for event in workflow.run_stream(task):
            if isinstance(event, WorkflowOutputEvent):
                for msg in event.data:
                    author = getattr(msg, "author_name", None) or "unknown"
                    text = getattr(msg, "text", "") or str(msg)
                    messages.append({
                        "agent": author,
                        "response": text,
                        "role": str(getattr(msg, "role", "assistant")),
                    })

        # Build structured agent_results
        for msg in messages:
            if msg.get("role", "").lower() != "user":
                result.agent_results.append(msg)

        # Final output = last assistant message
        assistant_msgs = [m for m in messages if m.get("role", "").lower() != "user"]
        result.final_output = assistant_msgs[-1]["response"] if assistant_msgs else ""
        result.status = "completed"
        result.metadata["agent_count"] = len(agents)
        result.metadata["message_count"] = len(messages)

    except Exception as exc:
        result.status = "failed"
        result.metadata["error"] = str(exc)
        logger.warning("SDK Sequential orchestration failed: %s", exc)

    result.elapsed_ms = round((time.time() - t0) * 1000, 2)
    return result


# ---------------------------------------------------------------------------
# SDK Orchestration: Concurrent
# ---------------------------------------------------------------------------


async def run_concurrent(
    agents: list[ChatAgent],
    task: str,
    **kwargs: Any,
) -> SDKOrchestrationResult:
    """Run a task on multiple agents concurrently using SDK ConcurrentBuilder.

    All agents receive the same input and process it in parallel.
    Results are aggregated into the final output.

    Args:
        agents: List of ChatAgents to run in parallel.
        task: The task description to process.

    Returns:
        SDKOrchestrationResult with parallel outputs merged.
    """
    result = SDKOrchestrationResult(pattern="sdk_concurrent", task=task)
    result.status = "running"
    t0 = time.time()

    if not agents:
        result.status = "failed"
        result.metadata["error"] = "No agents provided"
        result.elapsed_ms = round((time.time() - t0) * 1000, 2)
        return result

    try:
        workflow = ConcurrentBuilder().participants(agents).build()

        messages: list[dict[str, Any]] = []
        async for event in workflow.run_stream(task):
            if isinstance(event, WorkflowOutputEvent):
                for msg in event.data:
                    author = getattr(msg, "author_name", None) or "unknown"
                    text = getattr(msg, "text", "") or str(msg)
                    messages.append({
                        "agent": author,
                        "response": text,
                        "role": str(getattr(msg, "role", "assistant")),
                    })

        # Build per-agent results
        for msg in messages:
            if msg.get("role", "").lower() != "user":
                result.agent_results.append(msg)

        # Merge all assistant outputs
        parts = [
            f"[{m['agent']}]: {m['response']}"
            for m in result.agent_results
        ]
        result.final_output = "\n\n".join(parts) if parts else ""
        result.status = "completed"
        result.metadata["agent_count"] = len(agents)
        result.metadata["parallel_results"] = len(result.agent_results)

    except Exception as exc:
        result.status = "failed"
        result.metadata["error"] = str(exc)
        logger.warning("SDK Concurrent orchestration failed: %s", exc)

    result.elapsed_ms = round((time.time() - t0) * 1000, 2)
    return result


# ---------------------------------------------------------------------------
# SDK Orchestration: Handoff
# ---------------------------------------------------------------------------


async def run_handoff(
    participants: list[ChatAgent],
    task: str,
    start_agent: ChatAgent | None = None,
    max_turns: int = 5,
    **kwargs: Any,
) -> SDKOrchestrationResult:
    """Run a task using the SDK HandoffBuilder for dynamic agent delegation.

    The start agent receives the task first and can hand off to other
    participants. Uses a termination condition based on turn count or
    the agent signaling completion.

    Args:
        participants: All agents that can participate in the handoff.
        start_agent: Which agent receives the initial task (default: first).
        max_turns: Maximum conversation turns before termination.
        task: The task description.

    Returns:
        SDKOrchestrationResult with handoff chain and final output.
    """
    result = SDKOrchestrationResult(pattern="sdk_handoff", task=task)
    result.status = "running"
    t0 = time.time()

    if not participants:
        result.status = "failed"
        result.metadata["error"] = "No participants provided"
        result.elapsed_ms = round((time.time() - t0) * 1000, 2)
        return result

    initial = start_agent or participants[0]
    turn_count = 0

    def _termination(conversation: list) -> bool:
        nonlocal turn_count
        turn_count += 1
        if turn_count > max_turns:
            return True
        if conversation:
            last_text = getattr(conversation[-1], "text", "") or ""
            if "TASK_COMPLETE" in last_text or "complete" in last_text.lower()[-50:]:
                return True
        return False

    try:
        builder = HandoffBuilder(
            name="hirewire_handoff",
            participants=participants,
        )
        builder = builder.with_start_agent(initial)
        builder = builder.with_termination_condition(_termination)
        builder = builder.with_autonomous_mode()
        workflow = builder.build()

        messages: list[dict[str, Any]] = []
        async for event in workflow.run_stream(task):
            if isinstance(event, WorkflowOutputEvent):
                for msg in event.data:
                    author = getattr(msg, "author_name", None) or "unknown"
                    text = getattr(msg, "text", "") or str(msg)
                    messages.append({
                        "agent": author,
                        "response": text,
                        "role": str(getattr(msg, "role", "assistant")),
                    })

        for msg in messages:
            if msg.get("role", "").lower() != "user":
                result.agent_results.append(msg)

        result.final_output = result.agent_results[-1]["response"] if result.agent_results else ""
        result.status = "completed"
        result.metadata["agent_count"] = len(participants)
        result.metadata["turns"] = turn_count
        result.metadata["start_agent"] = initial.name

    except Exception as exc:
        result.status = "failed"
        result.metadata["error"] = str(exc)
        logger.warning("SDK Handoff orchestration failed: %s", exc)

    result.elapsed_ms = round((time.time() - t0) * 1000, 2)
    return result


# ---------------------------------------------------------------------------
# Unified SDK Orchestrator class
# ---------------------------------------------------------------------------


class SDKOrchestrator:
    """Unified orchestrator that routes tasks through SDK workflow builders.

    Provides a single interface for all three SDK patterns (Sequential,
    Concurrent, Handoff), with automatic fallback to HireWire's native
    orchestrator if the SDK is unavailable.

    Usage::

        orch = SDKOrchestrator()
        result = await orch.run("Research AI trends", pattern="sequential")
        result = await orch.run("Analyze from 3 perspectives", pattern="concurrent")
        result = await orch.run("CEO delegates to builder", pattern="handoff")
    """

    def __init__(self, chat_client: Any = None) -> None:
        self._agents = get_hirewire_sdk_agents(chat_client)
        self._history: list[SDKOrchestrationResult] = []
        self._sdk_available = _check_sdk_available()

    @property
    def sdk_available(self) -> bool:
        return self._sdk_available

    @property
    def agents(self) -> dict[str, ChatAgent]:
        return dict(self._agents)

    @property
    def history(self) -> list[SDKOrchestrationResult]:
        return list(self._history)

    async def run(
        self,
        task: str,
        pattern: str = "sequential",
        agents: list[str] | None = None,
        **kwargs: Any,
    ) -> SDKOrchestrationResult:
        """Run a task through the specified SDK orchestration pattern.

        Args:
            task: Task description to process.
            pattern: One of 'sequential', 'concurrent', 'handoff'.
            agents: Optional list of agent role names to include.
                   Defaults to pattern-appropriate selection.

        Returns:
            SDKOrchestrationResult with execution details.
        """
        selected = self._select_agents(agents, pattern)

        if not self._sdk_available:
            return await self._fallback_run(task, pattern, selected)

        if pattern == "sequential":
            result = await run_sequential(selected, task, **kwargs)
        elif pattern == "concurrent":
            result = await run_concurrent(selected, task, **kwargs)
        elif pattern == "handoff":
            result = await run_handoff(selected, task, **kwargs)
        else:
            result = SDKOrchestrationResult(
                pattern=pattern, task=task, status="failed",
                metadata={"error": f"Unknown pattern: {pattern}"},
            )

        self._history.append(result)
        return result

    def _select_agents(
        self, names: list[str] | None, pattern: str,
    ) -> list[ChatAgent]:
        """Select agents by role name, or use defaults for the pattern."""
        if names:
            return [
                self._agents[n] for n in names if n in self._agents
            ]

        # Pattern-specific defaults
        if pattern == "sequential":
            return [
                self._agents["research"],
                self._agents["builder"],
            ]
        elif pattern == "concurrent":
            return [
                self._agents["research"],
                self._agents["analyst"],
                self._agents["builder"],
            ]
        elif pattern == "handoff":
            return [
                self._agents["ceo"],
                self._agents["builder"],
                self._agents["research"],
            ]
        return list(self._agents.values())

    async def _fallback_run(
        self,
        task: str,
        pattern: str,
        agents: list[ChatAgent],
    ) -> SDKOrchestrationResult:
        """Fallback to HireWire's native orchestrator if SDK is unavailable."""
        from src.framework.agent import AgentFrameworkAgent
        from src.framework.orchestrator import (
            SequentialOrchestrator,
            ConcurrentOrchestrator,
            HandoffOrchestrator,
        )

        client = get_chat_client()
        native_agents = [
            AgentFrameworkAgent(
                name=a.name,
                description=getattr(a, "description", a.name),
                instructions=getattr(a, "instructions", ""),
                chat_client=client,
            )
            for a in agents
        ]

        if pattern == "sequential":
            orch = SequentialOrchestrator(native_agents)
        elif pattern == "concurrent":
            orch = ConcurrentOrchestrator(native_agents)
        elif pattern == "handoff":
            orch = HandoffOrchestrator(
                primary=native_agents[0],
                specialists=native_agents[1:],
            )
        else:
            return SDKOrchestrationResult(
                pattern=f"fallback_{pattern}", task=task, status="failed",
                metadata={"error": f"Unknown pattern: {pattern}"},
            )

        native_result = await orch.run(task)
        return SDKOrchestrationResult(
            pattern=f"fallback_{pattern}",
            task=task,
            status=native_result.status,
            agent_results=native_result.agent_results,
            final_output=native_result.final_output,
            elapsed_ms=native_result.elapsed_ms,
            metadata={**native_result.metadata, "fallback": True},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_sdk_available() -> bool:
    """Check if the Agent Framework SDK is properly installed."""
    try:
        from agent_framework import (
            ChatAgent,
            SequentialBuilder,
            ConcurrentBuilder,
            HandoffBuilder,
        )
        return True
    except ImportError:
        return False


def get_sdk_info() -> dict[str, Any]:
    """Return information about the SDK installation."""
    info: dict[str, Any] = {"available": _check_sdk_available()}
    try:
        import agent_framework
        info["version"] = getattr(agent_framework, "__version__", "unknown")
        info["patterns"] = ["sequential", "concurrent", "handoff"]
        info["features"] = [
            "SequentialBuilder",
            "ConcurrentBuilder",
            "HandoffBuilder",
            "ChatAgent",
            "MCPStdioTool",
            "MCPStreamableHTTPTool",
            "WorkflowOutputEvent",
        ]
    except ImportError:
        info["version"] = "not installed"
    return info
