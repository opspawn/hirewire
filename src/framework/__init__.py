"""Microsoft Agent Framework integration layer for AgentOS.

Provides Agent Framework-compatible abstractions that wrap AgentOS's
existing agent system, enabling multi-agent orchestration patterns
aligned with Microsoft's Agent Framework SDK.

Key components:
- AgentFrameworkAgent: Base agent class with MCP tool support
- Orchestrator: Sequential, concurrent, and handoff patterns
- MCPToolDescriptor: MCP-compatible tool definitions
- A2A: Agent-to-Agent protocol layer for inter-agent communication
"""

from src.framework.agent import AgentFrameworkAgent, AgentThread, AgentMessage
from src.framework.orchestrator import (
    Orchestrator,
    SequentialOrchestrator,
    ConcurrentOrchestrator,
    HandoffOrchestrator,
    OrchestratorResult,
)
from src.framework.mcp_tools import MCPToolDescriptor, MCPToolRegistry
from src.framework.a2a import A2AAgentCard, A2AServer, A2AClient

__all__ = [
    "AgentFrameworkAgent",
    "AgentThread",
    "AgentMessage",
    "Orchestrator",
    "SequentialOrchestrator",
    "ConcurrentOrchestrator",
    "HandoffOrchestrator",
    "OrchestratorResult",
    "MCPToolDescriptor",
    "MCPToolRegistry",
    "A2AAgentCard",
    "A2AServer",
    "A2AClient",
]
