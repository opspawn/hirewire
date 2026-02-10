"""Azure Cosmos DB persistence layer for AgentOS.

Provides a cloud-native persistence backend using Azure Cosmos DB (NoSQL API).
Creates a database ``agentos`` with containers for agents, jobs, and payments.

Environment variables:
    COSMOS_ENDPOINT — Cosmos DB account endpoint
    COSMOS_KEY      — Primary or secondary access key
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any


class CosmosDBStore:
    """Azure Cosmos DB persistence for agents, jobs, and payments.

    Uses the synchronous ``azure.cosmos`` SDK.  Containers are created on
    first access with ``/id`` as the partition key (suitable for small-medium
    workloads where cross-partition queries are acceptable).
    """

    DATABASE_NAME = "agentos"
    CONTAINERS = {
        "agents": "/id",
        "jobs": "/id",
        "payments": "/id",
    }

    def __init__(
        self,
        endpoint: str | None = None,
        key: str | None = None,
    ) -> None:
        self.endpoint = endpoint or os.environ.get("COSMOS_ENDPOINT", "")
        self.key = key or os.environ.get("COSMOS_KEY", "")
        self._client: Any = None
        self._database: Any = None
        self._containers: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        if self._client is None:
            from azure.cosmos import CosmosClient

            self._client = CosmosClient(self.endpoint, credential=self.key)
        return self._client

    @property
    def database(self) -> Any:
        if self._database is None:
            self._database = self.client.create_database_if_not_exists(
                id=self.DATABASE_NAME,
            )
        return self._database

    def _container(self, name: str) -> Any:
        """Return (and lazily create) a container proxy."""
        if name not in self._containers:
            from azure.cosmos import PartitionKey

            pk = self.CONTAINERS[name]
            self._containers[name] = self.database.create_container_if_not_exists(
                id=name,
                partition_key=PartitionKey(path=pk),
            )
        return self._containers[name]

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    def save_agent(self, agent: dict[str, Any]) -> dict[str, Any]:
        """Upsert an agent document.

        The dict **must** contain an ``id`` key.  Additional fields such as
        ``name``, ``description``, ``skills``, and ``pricing`` are stored
        as-is.
        """
        agent.setdefault("id", str(uuid.uuid4()))
        agent.setdefault("created_at", time.time())
        agent.setdefault("updated_at", time.time())
        return self._container("agents").upsert_item(agent)

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        """Fetch a single agent by ID.  Returns *None* if not found."""
        try:
            return self._container("agents").read_item(
                item=agent_id, partition_key=agent_id
            )
        except Exception:
            return None

    def list_agents(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return up to *limit* agent documents."""
        query = "SELECT * FROM c ORDER BY c.created_at DESC OFFSET 0 LIMIT @limit"
        return list(
            self._container("agents").query_items(
                query=query,
                parameters=[{"name": "@limit", "value": limit}],
                enable_cross_partition_query=True,
            )
        )

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def save_job(self, job: dict[str, Any]) -> dict[str, Any]:
        """Upsert a job document."""
        job.setdefault("id", str(uuid.uuid4()))
        job.setdefault("status", "pending")
        job.setdefault("created_at", time.time())
        job.setdefault("updated_at", time.time())
        return self._container("jobs").upsert_item(job)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        try:
            return self._container("jobs").read_item(
                item=job_id, partition_key=job_id
            )
        except Exception:
            return None

    def list_jobs(
        self,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List jobs, optionally filtered by status."""
        if status:
            query = "SELECT * FROM c WHERE c.status = @status ORDER BY c.created_at DESC OFFSET 0 LIMIT @limit"
            params = [
                {"name": "@status", "value": status},
                {"name": "@limit", "value": limit},
            ]
        else:
            query = "SELECT * FROM c ORDER BY c.created_at DESC OFFSET 0 LIMIT @limit"
            params = [{"name": "@limit", "value": limit}]

        return list(
            self._container("jobs").query_items(
                query=query,
                parameters=params,
                enable_cross_partition_query=True,
            )
        )

    # ------------------------------------------------------------------
    # Payments
    # ------------------------------------------------------------------

    def save_payment(self, payment: dict[str, Any]) -> dict[str, Any]:
        """Upsert a payment record."""
        payment.setdefault("id", str(uuid.uuid4()))
        payment.setdefault("status", "pending")
        payment.setdefault("created_at", time.time())
        return self._container("payments").upsert_item(payment)

    def get_payment(self, payment_id: str) -> dict[str, Any] | None:
        try:
            return self._container("payments").read_item(
                item=payment_id, partition_key=payment_id
            )
        except Exception:
            return None

    def list_payments(self, limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT * FROM c ORDER BY c.created_at DESC OFFSET 0 LIMIT @limit"
        return list(
            self._container("payments").query_items(
                query=query,
                parameters=[{"name": "@limit", "value": limit}],
                enable_cross_partition_query=True,
            )
        )

    # ------------------------------------------------------------------
    # Connectivity check
    # ------------------------------------------------------------------

    def check_connection(self) -> dict[str, Any]:
        """Verify Cosmos DB connectivity by listing databases."""
        try:
            dbs = list(self.client.list_databases())
            return {
                "connected": True,
                "databases": len(dbs),
                "endpoint": self.endpoint,
            }
        except Exception as exc:
            return {"connected": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_store: CosmosDBStore | None = None


def get_cosmos_store(**kwargs: Any) -> CosmosDBStore:
    """Return a cached :class:`CosmosDBStore` singleton."""
    global _store
    if _store is None:
        _store = CosmosDBStore(**kwargs)
    return _store


def cosmos_available() -> bool:
    """Return *True* if Cosmos DB environment variables are configured."""
    return bool(
        os.environ.get("COSMOS_ENDPOINT")
        and os.environ.get("COSMOS_KEY")
    )
