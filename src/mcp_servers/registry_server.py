"""MCP Server for Agent Registry - Discovery and capability matching.

Provides tools for discovering agents, registering capabilities,
and matching tasks to the best available agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool
from pydantic import BaseModel


@dataclass
class AgentCard:
    """Describes a registered agent's capabilities."""

    name: str
    description: str
    skills: list[str]
    price_per_call: str = "$0.00"
    endpoint: str = ""
    protocol: str = "a2a"
    payment: str = "x402"
    is_external: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    """In-memory registry of available agents."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentCard] = {}

    def register(self, card: AgentCard) -> None:
        """Register an agent in the registry."""
        self._agents[card.name] = card

    def unregister(self, name: str) -> bool:
        """Remove an agent from the registry."""
        return self._agents.pop(name, None) is not None

    def get(self, name: str) -> AgentCard | None:
        """Get an agent by name."""
        return self._agents.get(name)

    def search(self, capability: str, max_price: float | None = None) -> list[AgentCard]:
        """Search for agents matching a capability."""
        results = []
        capability_lower = capability.lower()
        for agent in self._agents.values():
            # Match against name, description, or skills
            if (
                capability_lower in agent.name.lower()
                or capability_lower in agent.description.lower()
                or any(capability_lower in s.lower() for s in agent.skills)
            ):
                if max_price is not None:
                    price = float(agent.price_per_call.replace("$", ""))
                    if price > max_price:
                        continue
                results.append(agent)
        return results

    def list_all(self) -> list[AgentCard]:
        """List all registered agents."""
        return list(self._agents.values())

    def to_summary(self) -> str:
        """Generate a human-readable summary for LLM context."""
        if not self._agents:
            return "No agents registered in the marketplace."
        lines = ["Available agents in the marketplace:"]
        for agent in self._agents.values():
            skills = ", ".join(agent.skills)
            lines.append(
                f"- {agent.name}: {agent.description} "
                f"(skills: {skills}, price: {agent.price_per_call})"
            )
        return "\n".join(lines)


# Global registry instance
registry = AgentRegistry()

# Pre-register internal agents
registry.register(AgentCard(
    name="builder",
    description="Writes code, runs tests, and deploys services",
    skills=["code", "testing", "deployment", "github"],
    price_per_call="$0.00",
    endpoint="internal://builder",
    protocol="internal",
    payment="none",
))

registry.register(AgentCard(
    name="research",
    description="Searches the web, analyzes data, produces reports",
    skills=["search", "analysis", "reports", "market-research"],
    price_per_call="$0.00",
    endpoint="internal://research",
    protocol="internal",
    payment="none",
))

# Pre-register the external mock designer agent
registry.register(AgentCard(
    name="designer-ext-001",
    description="Creates professional UI/UX designs, mockups, and design specifications",
    skills=["design", "ui", "ux", "mockup", "landing-page", "branding", "prototyping"],
    price_per_call="$0.05",
    endpoint="http://127.0.0.1:9100",
    protocol="a2a",
    payment="x402",
    is_external=True,
    metadata={
        "provider": "DesignStudio AI",
        "rating": 4.8,
        "tasks_completed": 142,
    },
))


def create_registry_mcp_server() -> Server:
    """Create the MCP server for agent registry operations."""
    server = Server("agent-registry")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="discover_agents",
                description="Search for agents by capability. Returns matching agents with skills and pricing.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "capability": {
                            "type": "string",
                            "description": "The capability to search for (e.g., 'screenshot', 'code', 'search')",
                        },
                        "max_price": {
                            "type": "number",
                            "description": "Maximum price per call in USD (optional)",
                        },
                    },
                    "required": ["capability"],
                },
            ),
            Tool(
                name="register_agent",
                description="Register a new agent in the marketplace with its capabilities and pricing.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Agent name"},
                        "description": {"type": "string", "description": "What the agent does"},
                        "skills": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of capabilities",
                        },
                        "price_per_call": {"type": "string", "description": "Price per call (e.g., '$0.01')"},
                        "endpoint": {"type": "string", "description": "A2A endpoint URL"},
                    },
                    "required": ["name", "description", "skills"],
                },
            ),
            Tool(
                name="list_agents",
                description="List all registered agents in the marketplace.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="discover_external_agents",
                description="Search for external (hireable) agents by capability. Returns agents available for hire with pricing and A2A endpoints.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "capability": {
                            "type": "string",
                            "description": "The capability to search for (e.g., 'design', 'screenshot')",
                        },
                        "max_price": {
                            "type": "number",
                            "description": "Maximum price per call in USD (optional)",
                        },
                    },
                    "required": ["capability"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "discover_agents":
            agents = registry.search(
                arguments["capability"],
                arguments.get("max_price"),
            )
            result = [asdict(a) for a in agents]
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        if name == "register_agent":
            card = AgentCard(
                name=arguments["name"],
                description=arguments["description"],
                skills=arguments.get("skills", []),
                price_per_call=arguments.get("price_per_call", "$0.00"),
                endpoint=arguments.get("endpoint", ""),
            )
            registry.register(card)
            return [TextContent(type="text", text=f"Agent '{card.name}' registered successfully")]

        if name == "list_agents":
            result = [asdict(a) for a in registry.list_all()]
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        if name == "discover_external_agents":
            all_matches = registry.search(
                arguments["capability"],
                arguments.get("max_price"),
            )
            external = [a for a in all_matches if a.is_external]
            result = [asdict(a) for a in external]
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server
