"""Tests for the Cosmos DB persistence layer.

All Cosmos DB interactions are mocked so tests run without credentials.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

os.environ.setdefault("MODEL_PROVIDER", "mock")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _cosmos_env(monkeypatch):
    """Set Cosmos DB environment variables for tests."""
    monkeypatch.setenv("COSMOS_ENDPOINT", "https://test-cosmos.documents.azure.com:443/")
    monkeypatch.setenv("COSMOS_KEY", "test-cosmos-key-12345==")
    import src.persistence.cosmos as mod
    mod._store = None
    yield
    mod._store = None


@pytest.fixture()
def mock_store(_cosmos_env):
    """Return a CosmosDBStore with a fully-mocked Cosmos client."""
    from src.persistence.cosmos import CosmosDBStore

    store = CosmosDBStore()

    # Mock the underlying Cosmos client chain
    mock_client = MagicMock()
    mock_database = MagicMock()
    mock_container = MagicMock()

    mock_client.create_database_if_not_exists.return_value = mock_database
    mock_database.create_container_if_not_exists.return_value = mock_container

    # upsert_item returns the item passed in
    mock_container.upsert_item.side_effect = lambda item: item
    # read_item returns the item as-is
    mock_container.read_item.side_effect = lambda item, partition_key: {
        "id": item, "name": "test"
    }
    # query_items returns an empty list by default
    mock_container.query_items.return_value = iter([])

    store._client = mock_client
    store._database = mock_database
    store._containers = {
        "agents": mock_container,
        "jobs": mock_container,
        "payments": mock_container,
    }
    return store, mock_container


# ---------------------------------------------------------------------------
# Tests — Agent CRUD
# ---------------------------------------------------------------------------


class TestAgentCRUD:
    def test_save_agent(self, mock_store):
        store, container = mock_store
        agent = {"id": "agent-1", "name": "Builder", "skills": ["code"]}
        result = store.save_agent(agent)
        assert result["id"] == "agent-1"
        container.upsert_item.assert_called_once()

    def test_save_agent_auto_id(self, mock_store):
        store, container = mock_store
        agent = {"name": "Research", "skills": ["search"]}
        result = store.save_agent(agent)
        assert "id" in result
        assert result["name"] == "Research"

    def test_get_agent(self, mock_store):
        store, container = mock_store
        result = store.get_agent("agent-1")
        assert result is not None
        assert result["id"] == "agent-1"
        container.read_item.assert_called_once_with(
            item="agent-1", partition_key="agent-1"
        )

    def test_get_agent_not_found(self, mock_store):
        store, container = mock_store
        container.read_item.side_effect = Exception("Not found")
        result = store.get_agent("nonexistent")
        assert result is None

    def test_list_agents(self, mock_store):
        store, container = mock_store
        container.query_items.return_value = iter([
            {"id": "a1", "name": "A"},
            {"id": "a2", "name": "B"},
        ])
        agents = store.list_agents()
        assert len(agents) == 2


# ---------------------------------------------------------------------------
# Tests — Job CRUD
# ---------------------------------------------------------------------------


class TestJobCRUD:
    def test_save_job(self, mock_store):
        store, container = mock_store
        job = {"id": "job-1", "description": "Build API", "status": "running"}
        result = store.save_job(job)
        assert result["id"] == "job-1"
        assert result["status"] == "running"

    def test_save_job_defaults(self, mock_store):
        store, container = mock_store
        job = {"description": "Test job"}
        result = store.save_job(job)
        assert "id" in result
        assert result["status"] == "pending"
        assert "created_at" in result

    def test_get_job(self, mock_store):
        store, container = mock_store
        result = store.get_job("job-1")
        assert result is not None

    def test_get_job_not_found(self, mock_store):
        store, container = mock_store
        container.read_item.side_effect = Exception("Not found")
        assert store.get_job("missing") is None

    def test_list_jobs_no_filter(self, mock_store):
        store, container = mock_store
        container.query_items.return_value = iter([{"id": "j1"}])
        jobs = store.list_jobs()
        assert len(jobs) == 1

    def test_list_jobs_with_status(self, mock_store):
        store, container = mock_store
        container.query_items.return_value = iter([])
        jobs = store.list_jobs(status="completed")
        assert len(jobs) == 0
        # Verify query used status filter
        call_args = container.query_items.call_args
        assert "@status" in call_args.kwargs.get("query", call_args[1].get("query", ""))


# ---------------------------------------------------------------------------
# Tests — Payment CRUD
# ---------------------------------------------------------------------------


class TestPaymentCRUD:
    def test_save_payment(self, mock_store):
        store, container = mock_store
        payment = {"id": "pay-1", "amount": 5.0, "from": "ceo", "to": "builder"}
        result = store.save_payment(payment)
        assert result["id"] == "pay-1"

    def test_save_payment_defaults(self, mock_store):
        store, container = mock_store
        payment = {"amount": 10.0}
        result = store.save_payment(payment)
        assert result["status"] == "pending"
        assert "created_at" in result

    def test_list_payments(self, mock_store):
        store, container = mock_store
        container.query_items.return_value = iter([{"id": "p1"}, {"id": "p2"}])
        payments = store.list_payments()
        assert len(payments) == 2


# ---------------------------------------------------------------------------
# Tests — Connectivity
# ---------------------------------------------------------------------------


class TestConnectivity:
    def test_check_connection_success(self, mock_store):
        store, _ = mock_store
        store._client.list_databases.return_value = [{"id": "agentos"}]
        result = store.check_connection()
        assert result["connected"] is True
        assert result["databases"] == 1

    def test_check_connection_failure(self, mock_store):
        store, _ = mock_store
        store._client.list_databases.side_effect = Exception("Timeout")
        result = store.check_connection()
        assert result["connected"] is False
        assert "Timeout" in result["error"]


# ---------------------------------------------------------------------------
# Tests — Module helpers
# ---------------------------------------------------------------------------


class TestModuleHelpers:
    def test_cosmos_available_true(self, _cosmos_env):
        from src.persistence.cosmos import cosmos_available
        assert cosmos_available() is True

    def test_cosmos_available_false(self, monkeypatch):
        monkeypatch.delenv("COSMOS_ENDPOINT", raising=False)
        monkeypatch.delenv("COSMOS_KEY", raising=False)
        from src.persistence.cosmos import cosmos_available
        assert cosmos_available() is False

    def test_get_cosmos_store_singleton(self, _cosmos_env):
        from src.persistence.cosmos import get_cosmos_store
        s1 = get_cosmos_store()
        s2 = get_cosmos_store()
        assert s1 is s2
