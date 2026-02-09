"""Tests for the dashboard frontend serving (src/dashboard/index.html).

Verifies that:
- GET / serves the dashboard HTML
- /dashboard/index.html is accessible via StaticFiles mount
- GET /tasks (list) returns a list
- HTML contains expected elements
"""

from __future__ import annotations

import pytest
import httpx

from src.api.main import app


@pytest.fixture()
def client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


class TestDashboardServing:
    @pytest.mark.asyncio
    async def test_root_serves_html(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_root_contains_agentos(self, client):
        resp = await client.get("/")
        assert "AgentOS" in resp.text

    @pytest.mark.asyncio
    async def test_root_contains_dashboard_panels(self, client):
        resp = await client.get("/")
        assert "Task Feed" in resp.text
        assert "Agent Roster" in resp.text
        assert "Payment Log" in resp.text
        assert "System Stats" in resp.text

    @pytest.mark.asyncio
    async def test_root_contains_submit_form(self, client):
        resp = await client.get("/")
        assert "task-input" in resp.text
        assert "submit-btn" in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_static_mount(self, client):
        resp = await client.get("/dashboard/index.html")
        assert resp.status_code == 200
        assert "AgentOS" in resp.text

    @pytest.mark.asyncio
    async def test_root_has_auto_refresh(self, client):
        resp = await client.get("/")
        assert "setInterval" in resp.text
        assert "refreshAll" in resp.text


class TestListTasksEndpoint:
    @pytest.mark.asyncio
    async def test_list_tasks_returns_list(self, client):
        resp = await client.get("/tasks")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_list_tasks_returns_valid_items(self, client):
        resp = await client.get("/tasks")
        data = resp.json()
        assert isinstance(data, list)
        for item in data:
            assert "task_id" in item
            assert "description" in item
            assert "status" in item

    @pytest.mark.asyncio
    async def test_list_tasks_includes_submitted(self, client):
        before = await client.get("/tasks")
        count_before = len(before.json())
        await client.post("/tasks", json={"description": "test list endpoint", "budget": 1.0})
        resp = await client.get("/tasks")
        tasks = resp.json()
        assert len(tasks) >= count_before + 1
        assert any(t["description"] == "test list endpoint" for t in tasks)
