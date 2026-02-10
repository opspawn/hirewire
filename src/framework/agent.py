"""AgentFrameworkAgent — Microsoft Agent Framework-compatible agent abstraction.

Wraps AgentOS's existing ChatAgent system with an interface that aligns
with Microsoft's Agent Framework patterns:
- Named agents with instructions (system prompt)
- MCP-compatible tool definitions
- Thread/conversation state tracking
- Connected Agents pattern for inter-agent communication
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from agent_framework import ChatAgent, tool

from src.config import get_chat_client


@dataclass
class AgentMessage:
    """A message in an agent conversation thread."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_name: str = ""


@dataclass
class AgentThread:
    """Tracks conversation state for an agent session.

    Mirrors Microsoft Agent Framework's thread concept — a persistent
    conversation context that agents can reference across invocations.
    """

    thread_id: str = field(default_factory=lambda: f"thread_{uuid.uuid4().hex[:12]}")
    messages: list[AgentMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, agent_name: str = "", **metadata: Any) -> AgentMessage:
        msg = AgentMessage(role=role, content=content, agent_name=agent_name, metadata=metadata)
        self.messages.append(msg)
        return msg

    def get_history(self, max_messages: int | None = None) -> list[AgentMessage]:
        if max_messages is None:
            return list(self.messages)
        return list(self.messages[-max_messages:])

    def clear(self) -> None:
        self.messages.clear()


# Type alias for tool execute functions
ToolExecuteFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class ToolDescriptor:
    """Describes a tool that an agent can use, compatible with MCP tool format."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    execute: ToolExecuteFn | None = None


class AgentFrameworkAgent:
    """Microsoft Agent Framework-compatible agent.

    Wraps the existing AgentOS ChatAgent with additional capabilities:
    - Tool management via MCP-compatible descriptors
    - Thread/conversation state tracking
    - Connected Agents pattern for delegation
    - Structured invoke() method for task processing
    """

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        tools: list[ToolDescriptor] | None = None,
        chat_client: Any = None,
        model: str = "default",
    ) -> None:
        self.name = name
        self.description = description
        self.instructions = instructions
        self.model = model
        self._tools: dict[str, ToolDescriptor] = {}
        self._connected_agents: dict[str, AgentFrameworkAgent] = {}
        self._threads: dict[str, AgentThread] = {}
        self._invoke_count: int = 0

        if tools:
            for t in tools:
                self._tools[t.name] = t

        # Create underlying ChatAgent
        self._chat_client = chat_client or get_chat_client()
        self._chat_agent = self._build_chat_agent()

    def _build_chat_agent(self) -> ChatAgent:
        """Build the underlying ChatAgent with registered tools."""
        agent_tools = []
        for td in self._tools.values():
            if td.execute is not None:
                # Wrap the execute function as an agent_framework tool
                agent_tools.append(self._make_agent_tool(td))

        return ChatAgent(
            chat_client=self._chat_client,
            name=self.name,
            description=self.description,
            instructions=self.instructions,
            tools=agent_tools,
        )

    @staticmethod
    def _make_agent_tool(td: ToolDescriptor) -> Any:
        """Convert a ToolDescriptor into an agent_framework-compatible tool."""
        execute_fn = td.execute

        @tool(name=td.name, description=td.description)
        async def wrapper(**kwargs: Any) -> dict[str, Any]:
            return await execute_fn(kwargs)

        return wrapper

    # --- Tool Management ---

    def add_tool(self, descriptor: ToolDescriptor) -> None:
        """Register a tool with this agent."""
        self._tools[descriptor.name] = descriptor
        self._chat_agent = self._build_chat_agent()

    def remove_tool(self, name: str) -> bool:
        """Remove a tool by name."""
        removed = self._tools.pop(name, None) is not None
        if removed:
            self._chat_agent = self._build_chat_agent()
        return removed

    def list_tools(self) -> list[ToolDescriptor]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> ToolDescriptor | None:
        """Get a tool by name."""
        return self._tools.get(name)

    # --- Connected Agents (Microsoft Agent Framework pattern) ---

    def connect_agent(self, agent: AgentFrameworkAgent) -> None:
        """Connect another agent for delegation/handoff."""
        self._connected_agents[agent.name] = agent

    def disconnect_agent(self, name: str) -> bool:
        """Disconnect a previously connected agent."""
        return self._connected_agents.pop(name, None) is not None

    def get_connected_agents(self) -> list[AgentFrameworkAgent]:
        """List all connected agents."""
        return list(self._connected_agents.values())

    def get_connected_agent(self, name: str) -> AgentFrameworkAgent | None:
        """Get a connected agent by name."""
        return self._connected_agents.get(name)

    # --- Thread Management ---

    def create_thread(self, **metadata: Any) -> AgentThread:
        """Create a new conversation thread."""
        thread = AgentThread(metadata=metadata)
        self._threads[thread.thread_id] = thread
        return thread

    def get_thread(self, thread_id: str) -> AgentThread | None:
        """Get an existing thread by ID."""
        return self._threads.get(thread_id)

    def list_threads(self) -> list[AgentThread]:
        """List all threads."""
        return list(self._threads.values())

    # --- Core Invocation ---

    async def invoke(
        self,
        task: str,
        thread: AgentThread | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a task and return structured results.

        This is the primary entry point, matching Microsoft Agent Framework's
        invoke pattern. The agent processes the task using its instructions
        and tools, maintaining conversation state in the thread.

        Args:
            task: The task description or user message.
            thread: Optional thread for conversation context. Created if None.
            context: Optional additional context dict.

        Returns:
            Dict with agent response, thread_id, and metadata.
        """
        self._invoke_count += 1

        if thread is None:
            thread = self.create_thread()

        # Add user message to thread
        thread.add_message("user", task, agent_name="user")

        # Build context-enriched task
        enriched_task = task
        if context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
            enriched_task = f"{task}\n\nContext:\n{context_str}"

        # Add connected agents info
        if self._connected_agents:
            agents_info = ", ".join(
                f"{a.name} ({a.description})" for a in self._connected_agents.values()
            )
            enriched_task += f"\n\nConnected agents available: {agents_info}"

        # Invoke the underlying ChatAgent
        t0 = time.time()
        result = await self._chat_agent.run(enriched_task)
        elapsed_ms = (time.time() - t0) * 1000

        # Extract response text
        response_text = self._extract_response(result)

        # Add assistant response to thread
        thread.add_message("assistant", response_text, agent_name=self.name)

        return {
            "agent": self.name,
            "response": response_text,
            "thread_id": thread.thread_id,
            "invoke_count": self._invoke_count,
            "elapsed_ms": round(elapsed_ms, 2),
            "tools_available": len(self._tools),
            "connected_agents": len(self._connected_agents),
        }

    async def delegate(
        self,
        agent_name: str,
        task: str,
        thread: AgentThread | None = None,
    ) -> dict[str, Any]:
        """Delegate a task to a connected agent.

        Part of the Connected Agents / Handoff pattern in Microsoft Agent Framework.

        Args:
            agent_name: Name of the connected agent to delegate to.
            task: Task to delegate.
            thread: Optional shared thread for context continuity.

        Returns:
            Dict with delegated agent's response.

        Raises:
            ValueError: If the named agent is not connected.
        """
        target = self._connected_agents.get(agent_name)
        if target is None:
            raise ValueError(
                f"Agent '{agent_name}' is not connected to '{self.name}'. "
                f"Connected: {list(self._connected_agents.keys())}"
            )

        if thread is not None:
            thread.add_message(
                "system",
                f"[Handoff] {self.name} → {agent_name}: {task}",
                agent_name=self.name,
            )

        result = await target.invoke(task, thread=thread)
        result["delegated_by"] = self.name
        return result

    @staticmethod
    def _extract_response(result: Any) -> str:
        """Extract text from a ChatAgent run result."""
        if hasattr(result, "get_outputs"):
            outputs = result.get_outputs()
            parts: list[str] = []
            for item in outputs:
                if isinstance(item, list):
                    for msg in item:
                        if hasattr(msg, "text") and msg.text:
                            parts.append(msg.text)
                elif hasattr(item, "text") and item.text:
                    parts.append(item.text)
                else:
                    parts.append(str(item))
            return "\n".join(parts) if parts else str(result)
        return str(result)

    # --- Metadata ---

    @property
    def agent_card(self) -> dict[str, Any]:
        """Generate an agent card for A2A discovery."""
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "tools": [
                {"name": t.name, "description": t.description}
                for t in self._tools.values()
            ],
            "connected_agents": [
                {"name": a.name, "description": a.description}
                for a in self._connected_agents.values()
            ],
            "capabilities": {
                "invoke": True,
                "delegate": bool(self._connected_agents),
                "threads": True,
            },
        }

    @property
    def invoke_count(self) -> int:
        return self._invoke_count

    def __repr__(self) -> str:
        return (
            f"AgentFrameworkAgent(name={self.name!r}, "
            f"tools={len(self._tools)}, "
            f"connected={len(self._connected_agents)})"
        )
