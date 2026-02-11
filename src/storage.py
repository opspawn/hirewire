"""SQLite persistence layer for HireWire.

Replaces in-memory dicts with durable storage using SQLite (WAL mode)
for tasks, payments, and agent registry. Uses aiosqlite for async access
with a synchronous fallback via sqlite3 for non-async call sites.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import aiosqlite

# Default database path (overridable via HIREWIRE_DB_PATH env var)
_DEFAULT_DB_PATH = Path(
    os.environ.get("HIREWIRE_DB_PATH", "")
    or str(Path(__file__).resolve().parent.parent / "data" / "hirewire.db")
)

# SQL schema
_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    workflow TEXT NOT NULL,
    budget_usd REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL,
    result TEXT  -- JSON blob
);

CREATE TABLE IF NOT EXISTS payments (
    tx_id TEXT PRIMARY KEY,
    from_agent TEXT NOT NULL,
    to_agent TEXT NOT NULL,
    amount_usdc REAL NOT NULL,
    task_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    tx_hash TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,  -- same as name
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    skills TEXT NOT NULL DEFAULT '[]',  -- JSON array
    price_per_call TEXT NOT NULL DEFAULT '$0.00',
    endpoint TEXT NOT NULL DEFAULT '',
    protocol TEXT NOT NULL DEFAULT 'a2a',
    payment TEXT NOT NULL DEFAULT 'x402',
    is_external INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}',  -- JSON blob
    registered_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS budgets (
    task_id TEXT PRIMARY KEY,
    allocated REAL NOT NULL,
    spent REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS tools (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    input_schema TEXT NOT NULL DEFAULT '{}',   -- JSON blob
    output_schema TEXT NOT NULL DEFAULT '{}',  -- JSON blob
    provider TEXT NOT NULL DEFAULT 'local',
    version TEXT NOT NULL DEFAULT '1.0.0',
    tags TEXT NOT NULL DEFAULT '[]',           -- JSON array
    registered_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,           -- 'task_completed', 'payment', etc.
    agent_id TEXT NOT NULL DEFAULT '',
    task_id TEXT NOT NULL DEFAULT '',
    task_type TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT '',    -- 'success', 'failure', 'timeout'
    cost_usdc REAL NOT NULL DEFAULT 0.0,
    latency_ms REAL NOT NULL DEFAULT 0.0,
    metadata TEXT NOT NULL DEFAULT '{}', -- JSON blob
    timestamp REAL NOT NULL
);
"""


class SQLiteStorage:
    """Unified SQLite storage for tasks, payments, and agent registry.

    Provides both synchronous and async methods. The synchronous methods
    use sqlite3 directly; async methods use aiosqlite.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or _DEFAULT_DB_PATH)
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a persistent synchronous connection with WAL mode.

        Reuses a single connection for the lifetime of the storage instance,
        avoiding the overhead of opening/closing connections per operation
        (~100ms per connect due to WAL + foreign key pragmas).
        """
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def save_task(
        self,
        task_id: str,
        description: str,
        workflow: str,
        budget_usd: float,
        status: str = "pending",
        created_at: float | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Insert or replace a task record."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO tasks
               (task_id, description, workflow, budget_usd, status, created_at, result)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                description,
                workflow,
                budget_usd,
                status,
                created_at or time.time(),
                json.dumps(result) if result is not None else None,
            ),
        )
        conn.commit()

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Retrieve a task by ID. Returns dict or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def list_tasks(self, status: str | None = None) -> list[dict[str, Any]]:
        """List all tasks, optionally filtered by status."""
        conn = self._get_conn()
        if status is not None:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status = ?", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tasks").fetchall()
        return [self._row_to_task(r) for r in rows]

    def update_task_status(
        self, task_id: str, status: str, result: dict[str, Any] | None = None
    ) -> None:
        """Update a task's status and optionally its result."""
        conn = self._get_conn()
        if result is not None:
            conn.execute(
                "UPDATE tasks SET status = ?, result = ? WHERE task_id = ?",
                (status, json.dumps(result), task_id),
            )
        else:
            conn.execute(
                "UPDATE tasks SET status = ? WHERE task_id = ?",
                (status, task_id),
            )
        conn.commit()

    def count_tasks(self, status: str | None = None) -> int:
        """Count tasks, optionally filtered by status."""
        conn = self._get_conn()
        if status is not None:
            row = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = ?", (status,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
        return row[0]

    def clear_tasks(self) -> None:
        """Delete all tasks (for testing)."""
        conn = self._get_conn()
        conn.execute("DELETE FROM tasks")
        conn.commit()

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> dict[str, Any]:
        result_raw = row["result"]
        return {
            "task_id": row["task_id"],
            "description": row["description"],
            "workflow": row["workflow"],
            "budget_usd": row["budget_usd"],
            "status": row["status"],
            "created_at": row["created_at"],
            "result": json.loads(result_raw) if result_raw else None,
        }

    # ------------------------------------------------------------------
    # Payments / Ledger
    # ------------------------------------------------------------------

    def save_payment(
        self,
        tx_id: str,
        from_agent: str,
        to_agent: str,
        amount_usdc: float,
        task_id: str,
        timestamp: float | None = None,
        status: str = "pending",
        tx_hash: str = "",
    ) -> None:
        """Insert a payment record."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO payments
               (tx_id, from_agent, to_agent, amount_usdc, task_id,
                timestamp, status, tx_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tx_id,
                from_agent,
                to_agent,
                amount_usdc,
                task_id,
                timestamp or time.time(),
                status,
                tx_hash,
            ),
        )
        conn.commit()

    def get_payments(self, task_id: str | None = None) -> list[dict[str, Any]]:
        """Get payment records, optionally filtered by task_id."""
        conn = self._get_conn()
        if task_id is not None:
            rows = conn.execute(
                "SELECT * FROM payments WHERE task_id = ?", (task_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM payments").fetchall()
        return [dict(r) for r in rows]

    def total_spent(self) -> float:
        """Total USDC spent (completed transactions)."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COALESCE(SUM(amount_usdc), 0) FROM payments WHERE status = 'completed'"
        ).fetchone()
        return row[0]

    def get_tx_count(self) -> int:
        """Get total number of transactions (for tx_id generation)."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM payments").fetchone()
        return row[0]

    def clear_payments(self) -> None:
        """Delete all payments (for testing)."""
        conn = self._get_conn()
        conn.execute("DELETE FROM payments")
        conn.commit()

    # ------------------------------------------------------------------
    # Budget helpers (stored as tasks metadata — we track via payments)
    # We use a simple approach: budgets are kept in a separate table-like
    # structure. For simplicity, we add a budgets table.
    # ------------------------------------------------------------------

    def save_budget(self, task_id: str, allocated: float, spent: float = 0.0) -> None:
        """Save or update a budget allocation."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO budgets (task_id, allocated, spent)
               VALUES (?, ?, ?)""",
            (task_id, allocated, spent),
        )
        conn.commit()

    def get_budget(self, task_id: str) -> dict[str, Any] | None:
        """Get budget for a task."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM budgets WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "task_id": row["task_id"],
            "allocated": row["allocated"],
            "spent": row["spent"],
            "remaining": row["allocated"] - row["spent"],
        }

    def update_budget_spent(self, task_id: str, additional_spent: float) -> None:
        """Add to the spent amount for a budget."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE budgets SET spent = spent + ? WHERE task_id = ?",
            (additional_spent, task_id),
        )
        conn.commit()

    def clear_budgets(self) -> None:
        """Delete all budgets (for testing)."""
        conn = self._get_conn()
        conn.execute("DELETE FROM budgets")
        conn.commit()

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    def save_agent(
        self,
        name: str,
        description: str,
        skills: list[str],
        price_per_call: str = "$0.00",
        endpoint: str = "",
        protocol: str = "a2a",
        payment: str = "x402",
        is_external: bool = False,
        metadata: dict[str, Any] | None = None,
        registered_at: float | None = None,
    ) -> None:
        """Register or update an agent."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO agents
               (agent_id, name, description, skills, price_per_call,
                endpoint, protocol, payment, is_external, metadata, registered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name,  # agent_id = name
                name,
                description,
                json.dumps(skills),
                price_per_call,
                endpoint,
                protocol,
                payment,
                1 if is_external else 0,
                json.dumps(metadata or {}),
                registered_at or time.time(),
            ),
        )
        conn.commit()

    def get_agent(self, name: str) -> dict[str, Any] | None:
        """Get an agent by name."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM agents WHERE agent_id = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_agent(row)

    def remove_agent(self, name: str) -> bool:
        """Remove an agent. Returns True if found and deleted."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM agents WHERE agent_id = ?", (name,))
        conn.commit()
        return cursor.rowcount > 0

    def list_agents(self) -> list[dict[str, Any]]:
        """List all agents."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM agents").fetchall()
        return [self._row_to_agent(r) for r in rows]

    def search_agents(
        self, capability: str, max_price: float | None = None
    ) -> list[dict[str, Any]]:
        """Search agents by capability (matches name, description, or skills)."""
        conn = self._get_conn()
        cap_lower = f"%{capability.lower()}%"
        rows = conn.execute(
            """SELECT * FROM agents
               WHERE LOWER(name) LIKE ?
                  OR LOWER(description) LIKE ?
                  OR LOWER(skills) LIKE ?""",
            (cap_lower, cap_lower, cap_lower),
        ).fetchall()
        results = [self._row_to_agent(r) for r in rows]
        if max_price is not None:
            results = [
                a for a in results
                if float(a["price_per_call"].replace("$", "")) <= max_price
            ]
        return results

    def clear_agents(self) -> None:
        """Delete all agents (for testing)."""
        conn = self._get_conn()
        conn.execute("DELETE FROM agents")
        conn.commit()

    @staticmethod
    def _row_to_agent(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "agent_id": row["agent_id"],
            "name": row["name"],
            "description": row["description"],
            "skills": json.loads(row["skills"]),
            "price_per_call": row["price_per_call"],
            "endpoint": row["endpoint"],
            "protocol": row["protocol"],
            "payment": row["payment"],
            "is_external": bool(row["is_external"]),
            "metadata": json.loads(row["metadata"]),
            "registered_at": row["registered_at"],
        }

    # ------------------------------------------------------------------
    # Tools (MCP Tool Server)
    # ------------------------------------------------------------------

    def save_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any] | None = None,
        provider: str = "local",
        version: str = "1.0.0",
        tags: list[str] | None = None,
        registered_at: float | None = None,
    ) -> None:
        """Register or update a tool."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO tools
               (name, description, input_schema, output_schema,
                provider, version, tags, registered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                description,
                json.dumps(input_schema),
                json.dumps(output_schema or {}),
                provider,
                version,
                json.dumps(tags or []),
                registered_at or time.time(),
            ),
        )
        conn.commit()

    def get_tool(self, name: str) -> dict[str, Any] | None:
        """Get a tool by name."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM tools WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_tool(row)

    def remove_tool(self, name: str) -> bool:
        """Remove a tool. Returns True if found and deleted."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM tools WHERE name = ?", (name,))
        conn.commit()
        return cursor.rowcount > 0

    def list_tools(self) -> list[dict[str, Any]]:
        """List all tools."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM tools").fetchall()
        return [self._row_to_tool(r) for r in rows]

    def clear_tools(self) -> None:
        """Delete all tools (for testing)."""
        conn = self._get_conn()
        conn.execute("DELETE FROM tools")
        conn.commit()

    @staticmethod
    def _row_to_tool(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "name": row["name"],
            "description": row["description"],
            "input_schema": json.loads(row["input_schema"]),
            "output_schema": json.loads(row["output_schema"]),
            "provider": row["provider"],
            "version": row["version"],
            "tags": json.loads(row["tags"]),
            "registered_at": row["registered_at"],
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def save_metric(
        self,
        event_type: str,
        agent_id: str = "",
        task_id: str = "",
        task_type: str = "",
        status: str = "",
        cost_usdc: float = 0.0,
        latency_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Insert a metrics event."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO metrics
               (event_type, agent_id, task_id, task_type, status,
                cost_usdc, latency_ms, metadata, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_type,
                agent_id,
                task_id,
                task_type,
                status,
                cost_usdc,
                latency_ms,
                json.dumps(metadata or {}),
                timestamp or time.time(),
            ),
        )
        conn.commit()

    def get_metrics(
        self,
        event_type: str | None = None,
        agent_id: str | None = None,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        """Query metrics with optional filters."""
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM metrics{where} ORDER BY timestamp DESC", params
        ).fetchall()
        return [self._row_to_metric(r) for r in rows]

    def clear_metrics(self) -> None:
        """Delete all metrics (for testing)."""
        conn = self._get_conn()
        conn.execute("DELETE FROM metrics")
        conn.commit()

    @staticmethod
    def _row_to_metric(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "event_type": row["event_type"],
            "agent_id": row["agent_id"],
            "task_id": row["task_id"],
            "task_type": row["task_type"],
            "status": row["status"],
            "cost_usdc": row["cost_usdc"],
            "latency_ms": row["latency_ms"],
            "metadata": json.loads(row["metadata"]),
            "timestamp": row["timestamp"],
        }

    # ------------------------------------------------------------------
    # Async wrappers (via aiosqlite)
    # ------------------------------------------------------------------

    async def async_save_task(self, **kwargs: Any) -> None:
        """Async version of save_task."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            created_at = kwargs.get("created_at") or time.time()
            result = kwargs.get("result")
            await db.execute(
                """INSERT OR REPLACE INTO tasks
                   (task_id, description, workflow, budget_usd, status, created_at, result)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    kwargs["task_id"],
                    kwargs["description"],
                    kwargs["workflow"],
                    kwargs["budget_usd"],
                    kwargs.get("status", "pending"),
                    created_at,
                    json.dumps(result) if result is not None else None,
                ),
            )
            await db.commit()

    async def async_update_task_status(
        self, task_id: str, status: str, result: dict[str, Any] | None = None
    ) -> None:
        """Async version of update_task_status."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            if result is not None:
                await db.execute(
                    "UPDATE tasks SET status = ?, result = ? WHERE task_id = ?",
                    (status, json.dumps(result), task_id),
                )
            else:
                await db.execute(
                    "UPDATE tasks SET status = ? WHERE task_id = ?",
                    (status, task_id),
                )
            await db.commit()

    async def async_get_task(self, task_id: str) -> dict[str, Any] | None:
        """Async version of get_task."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            cursor = await db.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            result_raw = row["result"]
            return {
                "task_id": row["task_id"],
                "description": row["description"],
                "workflow": row["workflow"],
                "budget_usd": row["budget_usd"],
                "status": row["status"],
                "created_at": row["created_at"],
                "result": json.loads(result_raw) if result_raw else None,
            }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear_all(self) -> None:
        """Clear all tables (for testing)."""
        self.clear_tasks()
        self.clear_payments()
        self.clear_budgets()
        self.clear_agents()
        self.clear_tools()
        self.clear_metrics()

    def close(self) -> None:
        """Close the persistent connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# Module-level singleton
_storage: SQLiteStorage | None = None


def get_storage(db_path: str | Path | None = None) -> SQLiteStorage:
    """Get or create the global storage instance."""
    global _storage
    if _storage is None:
        _storage = SQLiteStorage(db_path)
    return _storage


def reset_storage(db_path: str | Path | None = None) -> SQLiteStorage:
    """Reset the global storage instance (for testing)."""
    global _storage
    _storage = SQLiteStorage(db_path)
    return _storage
