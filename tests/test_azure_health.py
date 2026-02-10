"""Tests for the /health/azure endpoint."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("MODEL_PROVIDER", "mock")

from src.api.main import app  # noqa: E402

client = TestClient(app)


class TestAzureHealthEndpoint:
    """Test the /health/azure deep health check."""

    def test_health_azure_not_configured(self, monkeypatch):
        """When Azure env vars are missing, endpoint reports degraded."""
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)
        monkeypatch.delenv("COSMOS_ENDPOINT", raising=False)
        monkeypatch.delenv("COSMOS_KEY", raising=False)

        # Reset singletons
        import src.framework.azure_llm as azure_mod
        import src.persistence.cosmos as cosmos_mod
        azure_mod._provider = None
        cosmos_mod._store = None

        resp = client.get("/health/azure")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["services"]["azure_openai"]["connected"] is False
        assert data["services"]["cosmos_db"]["connected"] is False

    @patch("src.framework.azure_llm.get_azure_llm")
    @patch("src.persistence.cosmos.get_cosmos_store")
    def test_health_azure_all_connected(
        self, mock_cosmos_fn, mock_llm_fn, monkeypatch
    ):
        """When both services are reachable, endpoint reports healthy."""
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_KEY", "test-key")
        monkeypatch.setenv("COSMOS_ENDPOINT", "https://test.documents.azure.com/")
        monkeypatch.setenv("COSMOS_KEY", "test-key")

        mock_llm = MagicMock()
        mock_llm.check_connection.return_value = {"connected": True, "model": "gpt-4o"}
        mock_llm_fn.return_value = mock_llm

        mock_store = MagicMock()
        mock_store.check_connection.return_value = {"connected": True, "databases": 1}
        mock_cosmos_fn.return_value = mock_store

        resp = client.get("/health/azure")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["services"]["azure_openai"]["connected"] is True
        assert data["services"]["cosmos_db"]["connected"] is True
