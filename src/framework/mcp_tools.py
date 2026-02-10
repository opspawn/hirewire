"""MCP Tool integration for Microsoft Agent Framework.

Defines MCP-compatible tool descriptors and a registry that maps
existing AgentOS capabilities as framework tools. Each tool has:
- Name and description
- JSON Schema parameters
- Async execute function

This module bridges AgentOS's ToolDefinition/ToolRegistry system with
the Microsoft Agent Framework's MCP tool pattern.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from src.framework.agent import ToolDescriptor

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class MCPToolDescriptor:
    """MCP-compatible tool descriptor.

    Extends the basic ToolDescriptor with MCP protocol metadata:
    server name, version, tags, and invocation tracking.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    server: str = "agentos"
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    _execute: ToolHandler | None = field(default=None, repr=False)

    async def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool with given arguments."""
        if self._execute is None:
            return {"status": "ok", "echo": args, "tool": self.name}
        return await self._execute(args)

    def to_tool_descriptor(self) -> ToolDescriptor:
        """Convert to an AgentFrameworkAgent-compatible ToolDescriptor."""
        handler = self._execute

        async def _exec(a: dict[str, Any]) -> dict[str, Any]:
            return await self.execute(a)

        return ToolDescriptor(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            execute=_exec,
        )

    def to_mcp_dict(self) -> dict[str, Any]:
        """Serialize to MCP tools/list response format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
            "server": self.server,
            "version": self.version,
            "tags": self.tags,
        }


class MCPToolRegistry:
    """Registry for MCP-compatible tools.

    Manages tool registration, discovery, and invocation tracking.
    Maps existing AgentOS capabilities as MCP tools that can be
    attached to AgentFrameworkAgent instances.
    """

    def __init__(self) -> None:
        self._tools: dict[str, MCPToolDescriptor] = {}
        self._invocation_log: list[dict[str, Any]] = []

    def register(self, tool: MCPToolDescriptor) -> None:
        """Register an MCP tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry."""
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> MCPToolDescriptor | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def search(self, query: str) -> list[MCPToolDescriptor]:
        """Search tools by name, description, or tags."""
        query_lower = query.lower()
        results = []
        for t in self._tools.values():
            if (
                query_lower in t.name.lower()
                or query_lower in t.description.lower()
                or any(query_lower in tag.lower() for tag in t.tags)
            ):
                results.append(t)
        return results

    def search_by_tag(self, tag: str) -> list[MCPToolDescriptor]:
        """Search tools by a specific tag."""
        tag_lower = tag.lower()
        return [t for t in self._tools.values() if any(tag_lower == tt.lower() for tt in t.tags)]

    def list_all(self) -> list[MCPToolDescriptor]:
        """List all registered tools."""
        return list(self._tools.values())

    async def invoke(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Invoke a tool by name and track the invocation."""
        tool = self._tools.get(name)
        if tool is None:
            return {"status": "error", "error": f"Tool not found: {name}"}

        inv_id = f"mcp_inv_{uuid.uuid4().hex[:8]}"
        t0 = time.time()
        try:
            result = await tool.execute(args)
            elapsed_ms = round((time.time() - t0) * 1000, 2)
            log_entry = {
                "invocation_id": inv_id,
                "tool": name,
                "args": args,
                "result": result,
                "status": "completed",
                "elapsed_ms": elapsed_ms,
            }
            self._invocation_log.append(log_entry)
            return result
        except Exception as exc:
            elapsed_ms = round((time.time() - t0) * 1000, 2)
            log_entry = {
                "invocation_id": inv_id,
                "tool": name,
                "args": args,
                "status": "failed",
                "error": str(exc),
                "elapsed_ms": elapsed_ms,
            }
            self._invocation_log.append(log_entry)
            return {"status": "error", "error": str(exc)}

    def get_invocation_log(self) -> list[dict[str, Any]]:
        """Get the invocation history."""
        return list(self._invocation_log)

    def clear(self) -> None:
        """Clear all tools and invocation history."""
        self._tools.clear()
        self._invocation_log.clear()

    def to_mcp_list(self) -> list[dict[str, Any]]:
        """Serialize all tools to MCP tools/list format."""
        return [t.to_mcp_dict() for t in self._tools.values()]


# ---------------------------------------------------------------------------
# Pre-built MCP tools mapping AgentOS capabilities
# ---------------------------------------------------------------------------

async def _screenshot_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Take a screenshot of a URL (mock implementation)."""
    url = args.get("url", "")
    return {
        "status": "ok",
        "url": url,
        "screenshot_path": f"/tmp/screenshot_{uuid.uuid4().hex[:8]}.png",
        "width": args.get("width", 1280),
        "height": args.get("height", 720),
        "format": "png",
    }


async def _ai_analysis_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Analyze content using AI reasoning (mock implementation)."""
    content = args.get("content", "")
    analysis_type = args.get("type", "summary")
    return {
        "status": "ok",
        "analysis_type": analysis_type,
        "result": f"Analysis of: {content[:100]}",
        "confidence": 0.85,
        "insights": [f"Mock insight for {analysis_type} analysis"],
    }


async def _markdown_conversion_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Convert content to markdown (mock implementation)."""
    content = args.get("content", "")
    source_format = args.get("source_format", "html")
    return {
        "status": "ok",
        "source_format": source_format,
        "markdown": f"# Converted from {source_format}\n\n{content[:200]}",
        "length": len(content),
    }


async def _web_search_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Search the web for information (mock implementation)."""
    query = args.get("query", "")
    max_results = args.get("max_results", 5)
    return {
        "status": "ok",
        "query": query,
        "results": [
            {"title": f"Result {i+1} for: {query}", "url": f"https://example.com/{i+1}", "snippet": f"Mock result {i+1}"}
            for i in range(min(max_results, 5))
        ],
        "total_results": max_results,
    }


async def _file_operation_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Perform file operations (mock implementation)."""
    operation = args.get("operation", "read")
    path = args.get("path", "")
    return {
        "status": "ok",
        "operation": operation,
        "path": path,
        "result": f"Mock {operation} on {path}",
    }


async def _api_call_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Make an API call (mock implementation)."""
    url = args.get("url", "")
    method = args.get("method", "GET")
    return {
        "status": "ok",
        "url": url,
        "method": method,
        "response_code": 200,
        "body": {"message": f"Mock {method} response from {url}"},
    }


# Pre-built tool instances
SCREENSHOT_TOOL = MCPToolDescriptor(
    name="screenshot",
    description="Capture a screenshot of a web page at a given URL",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to screenshot"},
            "width": {"type": "integer", "description": "Viewport width (default 1280)"},
            "height": {"type": "integer", "description": "Viewport height (default 720)"},
        },
        "required": ["url"],
    },
    tags=["browser", "screenshot", "visual"],
    _execute=_screenshot_handler,
)

AI_ANALYSIS_TOOL = MCPToolDescriptor(
    name="ai_analysis",
    description="Analyze content using AI reasoning and produce structured insights",
    parameters={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Content to analyze"},
            "type": {"type": "string", "description": "Analysis type: summary, sentiment, extraction, comparison"},
        },
        "required": ["content"],
    },
    tags=["ai", "analysis", "reasoning"],
    _execute=_ai_analysis_handler,
)

MARKDOWN_CONVERSION_TOOL = MCPToolDescriptor(
    name="markdown_conversion",
    description="Convert content from various formats (HTML, JSON, plain text) to Markdown",
    parameters={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Content to convert"},
            "source_format": {"type": "string", "description": "Source format: html, json, text"},
        },
        "required": ["content"],
    },
    tags=["conversion", "markdown", "format"],
    _execute=_markdown_conversion_handler,
)

WEB_SEARCH_TOOL = MCPToolDescriptor(
    name="web_search",
    description="Search the web for information and return structured results",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Maximum number of results (default 5)"},
        },
        "required": ["query"],
    },
    tags=["search", "web", "research"],
    _execute=_web_search_handler,
)

FILE_OPERATION_TOOL = MCPToolDescriptor(
    name="file_operation",
    description="Perform file system operations: read, write, list, or delete files",
    parameters={
        "type": "object",
        "properties": {
            "operation": {"type": "string", "description": "Operation: read, write, list, delete"},
            "path": {"type": "string", "description": "File or directory path"},
            "content": {"type": "string", "description": "Content for write operations"},
        },
        "required": ["operation", "path"],
    },
    tags=["file", "filesystem", "io"],
    _execute=_file_operation_handler,
)

API_CALL_TOOL = MCPToolDescriptor(
    name="api_call",
    description="Make HTTP API calls to external services",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "API endpoint URL"},
            "method": {"type": "string", "description": "HTTP method: GET, POST, PUT, DELETE"},
            "body": {"type": "object", "description": "Request body (for POST/PUT)"},
            "headers": {"type": "object", "description": "Request headers"},
        },
        "required": ["url"],
    },
    tags=["api", "http", "integration"],
    _execute=_api_call_handler,
)

async def _code_execution_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Execute code in a sandboxed environment (mock implementation)."""
    language = args.get("language", "python")
    code = args.get("code", "")
    return {
        "status": "ok",
        "language": language,
        "output": f"Mock execution of {language} code ({len(code)} chars)",
        "exit_code": 0,
        "execution_time_ms": 42,
    }


async def _data_store_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Store or retrieve key-value data (mock implementation)."""
    operation = args.get("operation", "get")
    key = args.get("key", "")
    value = args.get("value")
    if operation == "set":
        return {"status": "ok", "operation": "set", "key": key, "stored": True}
    elif operation == "get":
        return {"status": "ok", "operation": "get", "key": key, "value": f"mock_value_for_{key}"}
    elif operation == "delete":
        return {"status": "ok", "operation": "delete", "key": key, "deleted": True}
    elif operation == "list":
        return {"status": "ok", "operation": "list", "keys": [f"key_{i}" for i in range(3)]}
    return {"status": "error", "error": f"Unknown operation: {operation}"}


CODE_EXECUTION_TOOL = MCPToolDescriptor(
    name="code_execution",
    description="Execute code in a sandboxed environment supporting Python, JavaScript, and shell",
    parameters={
        "type": "object",
        "properties": {
            "language": {"type": "string", "description": "Language: python, javascript, shell"},
            "code": {"type": "string", "description": "Code to execute"},
            "timeout_ms": {"type": "integer", "description": "Max execution time in ms (default 30000)"},
        },
        "required": ["language", "code"],
    },
    tags=["code", "execution", "sandbox"],
    _execute=_code_execution_handler,
)

DATA_STORE_TOOL = MCPToolDescriptor(
    name="data_store",
    description="Key-value data store for persisting agent state and sharing data between agents",
    parameters={
        "type": "object",
        "properties": {
            "operation": {"type": "string", "description": "Operation: get, set, delete, list"},
            "key": {"type": "string", "description": "Key to store/retrieve"},
            "value": {"description": "Value to store (for set operations)"},
        },
        "required": ["operation"],
    },
    tags=["storage", "data", "state"],
    _execute=_data_store_handler,
)


# All pre-built tools
BUILTIN_TOOLS = [
    SCREENSHOT_TOOL,
    AI_ANALYSIS_TOOL,
    MARKDOWN_CONVERSION_TOOL,
    WEB_SEARCH_TOOL,
    FILE_OPERATION_TOOL,
    API_CALL_TOOL,
    CODE_EXECUTION_TOOL,
    DATA_STORE_TOOL,
]


def create_default_registry() -> MCPToolRegistry:
    """Create an MCPToolRegistry pre-loaded with all built-in tools."""
    registry = MCPToolRegistry()
    for tool in BUILTIN_TOOLS:
        registry.register(tool)
    return registry
