"""A2A (Agent-to-Agent) Protocol Layer for Microsoft Agent Framework.

Provides agent-to-agent communication following Google A2A spec patterns:
- A2AAgentCard: Agent discovery metadata
- A2AServer: Expose an AgentFrameworkAgent as an A2A endpoint
- A2AClient: Discover and invoke remote A2A agents

This enables inter-agent communication for the hackathon demo,
where agents can discover each other and exchange tasks.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

import httpx

from src.framework.agent import AgentFrameworkAgent


@dataclass
class A2AAgentCard:
    """Agent card for A2A protocol discovery.

    Matches the Google A2A agent-card specification format.
    """

    name: str
    description: str
    url: str = ""
    version: str = "1.0.0"
    skills: list[str] = field(default_factory=list)
    protocols: list[str] = field(default_factory=lambda: ["a2a", "json-rpc-2.0"])
    authentication: dict[str, Any] = field(
        default_factory=lambda: {"schemes": ["none"]}
    )
    pricing: dict[str, Any] = field(
        default_factory=lambda: {"model": "free", "currency": "USDC"}
    )
    endpoints: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_agent(
        cls,
        agent: AgentFrameworkAgent,
        base_url: str = "http://localhost:8080",
    ) -> A2AAgentCard:
        """Generate an A2A agent card from an AgentFrameworkAgent."""
        skills = [t.name for t in agent.list_tools()]
        return cls(
            name=agent.name,
            description=agent.description,
            url=base_url,
            skills=skills,
            endpoints={
                "jsonrpc": f"{base_url}/a2a",
                "agent_card": f"{base_url}/.well-known/agent.json",
                "health": f"{base_url}/a2a/health",
            },
            metadata={
                "tools_count": len(agent.list_tools()),
                "connected_agents": len(agent.get_connected_agents()),
                "model": agent.model,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON responses."""
        return asdict(self)

    def matches_capability(self, capability: str) -> bool:
        """Check if this agent card matches a capability query."""
        cap_lower = capability.lower()
        return (
            cap_lower in self.name.lower()
            or cap_lower in self.description.lower()
            or any(cap_lower in s.lower() for s in self.skills)
        )


@dataclass
class A2ATask:
    """Represents a task in the A2A protocol."""

    task_id: str = field(default_factory=lambda: f"a2a_{uuid.uuid4().hex[:12]}")
    description: str = ""
    from_agent: str = ""
    to_agent: str = ""
    status: str = "pending"  # pending, running, completed, failed
    result: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


class A2AServer:
    """Exposes an AgentFrameworkAgent as an A2A protocol endpoint.

    Handles incoming A2A requests (tasks/send, tasks/get, agents/info)
    and routes them to the underlying agent.
    """

    def __init__(
        self,
        agent: AgentFrameworkAgent,
        base_url: str = "http://localhost:8080",
    ) -> None:
        self._agent = agent
        self._base_url = base_url
        self._card = A2AAgentCard.from_agent(agent, base_url)
        self._tasks: dict[str, A2ATask] = {}

    @property
    def agent_card(self) -> A2AAgentCard:
        return self._card

    def get_agent_card_dict(self) -> dict[str, Any]:
        """Get the agent card as a dictionary."""
        return self._card.to_dict()

    async def handle_task(
        self,
        description: str,
        from_agent: str = "anonymous",
    ) -> A2ATask:
        """Handle an incoming task request.

        Creates a task, invokes the agent, and returns the result.
        """
        task = A2ATask(
            description=description,
            from_agent=from_agent,
            to_agent=self._agent.name,
        )
        self._tasks[task.task_id] = task
        task.status = "running"

        try:
            result = await self._agent.invoke(
                description,
                context={"from_agent": from_agent, "protocol": "a2a"},
            )
            task.status = "completed"
            task.result = result
            task.completed_at = time.time()
        except Exception as exc:
            task.status = "failed"
            task.result = {"error": str(exc)}
            task.completed_at = time.time()

        return task

    def get_task(self, task_id: str) -> A2ATask | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[A2ATask]:
        """List all tasks."""
        return list(self._tasks.values())

    def dispatch_jsonrpc(self, request_body: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a JSON-RPC 2.0 request.

        Supported methods:
        - tasks/send: Submit a task to the agent
        - tasks/get: Check task status
        - agents/info: Get agent card
        """
        if not isinstance(request_body, dict):
            return self._error(-32600, "Invalid request")

        jsonrpc = request_body.get("jsonrpc")
        if jsonrpc != "2.0":
            return self._error(-32600, "Invalid JSON-RPC version")

        method = request_body.get("method", "")
        req_id = request_body.get("id")
        params = request_body.get("params", {})

        if method == "agents/info":
            return self._result(self.get_agent_card_dict(), req_id)

        if method == "tasks/get":
            task_id = params.get("task_id", "")
            task = self.get_task(task_id)
            if task is None:
                return self._result({"error": f"Task not found: {task_id}"}, req_id)
            return self._result(asdict(task), req_id)

        if method == "tasks/send":
            # Note: in a real server this would be async
            # For testing we return immediately with pending status
            description = params.get("description", "")
            from_agent = params.get("from_agent", "anonymous")
            task = A2ATask(
                description=description,
                from_agent=from_agent,
                to_agent=self._agent.name,
            )
            self._tasks[task.task_id] = task
            return self._result(asdict(task), req_id)

        return self._error(-32601, f"Method not found: {method}", req_id)

    @staticmethod
    def _result(data: Any, req_id: Any = None) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "result": data, "id": req_id}

    @staticmethod
    def _error(code: int, message: str, req_id: Any = None) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": req_id}


class A2AClient:
    """Client for discovering and invoking remote A2A agents.

    Connects to A2A endpoints, fetches agent cards, and sends tasks.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._discovered: dict[str, A2AAgentCard] = {}

    async def discover(self, base_url: str) -> A2AAgentCard | None:
        """Discover an agent by fetching its agent card.

        Args:
            base_url: The base URL of the A2A agent (e.g., http://agent:8080)

        Returns:
            A2AAgentCard if successful, None if unreachable.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{base_url}/.well-known/agent.json")
                resp.raise_for_status()
                data = resp.json()
                card = A2AAgentCard(
                    name=data.get("name", "unknown"),
                    description=data.get("description", ""),
                    url=data.get("url", base_url),
                    version=data.get("version", "1.0.0"),
                    skills=data.get("skills", []),
                    protocols=data.get("protocols", []),
                    authentication=data.get("authentication", {}),
                    pricing=data.get("pricing", {}),
                    endpoints=data.get("endpoints", {}),
                    metadata=data.get("metadata", {}),
                )
                self._discovered[card.name] = card
                return card
        except Exception:
            return None

    async def send_task(
        self,
        base_url: str,
        description: str,
        from_agent: str = "anonymous",
    ) -> dict[str, Any]:
        """Send a task to a remote A2A agent.

        Args:
            base_url: The base URL of the target agent.
            description: Task description.
            from_agent: Name of the sending agent.

        Returns:
            JSON-RPC response dict.
        """
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "description": description,
                "from_agent": from_agent,
            },
            "id": str(uuid.uuid4().hex[:8]),
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{base_url}/a2a", json=payload)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            return {"error": str(exc)}

    async def get_task_status(
        self,
        base_url: str,
        task_id: str,
    ) -> dict[str, Any]:
        """Check the status of a previously sent task.

        Args:
            base_url: The base URL of the target agent.
            task_id: The task ID to check.

        Returns:
            JSON-RPC response dict with task status.
        """
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"task_id": task_id},
            "id": str(uuid.uuid4().hex[:8]),
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{base_url}/a2a", json=payload)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            return {"error": str(exc)}

    def get_discovered(self) -> list[A2AAgentCard]:
        """List all discovered agents."""
        return list(self._discovered.values())

    def find_by_capability(self, capability: str) -> list[A2AAgentCard]:
        """Find discovered agents matching a capability."""
        return [c for c in self._discovered.values() if c.matches_capability(capability)]
