"""Tests for HITL API endpoints.

Tests cover all /approvals/* endpoints via FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.hitl import reset_approval_gate, get_approval_gate


@pytest.fixture(autouse=True)
def _reset_gate():
    """Reset approval gate between tests."""
    reset_approval_gate()
    yield
    reset_approval_gate()


@pytest.fixture
def client():
    return TestClient(app)


class TestApprovalEndpoints:
    def test_get_pending_empty(self, client):
        resp = client.get("/approvals/pending")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_stats(self, client):
        resp = client.get("/approvals/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] == 0
        assert data["pending"] == 0
        assert "cost_threshold" in data

    def test_approve_flow(self, client):
        # Create a pending approval
        gate = get_approval_gate()
        rid = gate.request_approval("test_action", cost_usdc=5.0)

        # Check pending list
        resp = client.get("/approvals/pending")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["request_id"] == rid

        # Check status
        resp = client.get(f"/approvals/{rid}/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

        # Approve
        resp = client.post(
            f"/approvals/{rid}/approve",
            json={"reviewer": "tester", "reason": "Test approved"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"
        assert resp.json()["reviewer"] == "tester"

        # Pending should be empty
        resp = client.get("/approvals/pending")
        assert resp.json() == []

    def test_reject_flow(self, client):
        gate = get_approval_gate()
        rid = gate.request_approval("test_action", cost_usdc=5.0)

        resp = client.post(
            f"/approvals/{rid}/reject",
            json={"reviewer": "tester", "reason": "Too costly"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"

    def test_approve_nonexistent(self, client):
        resp = client.post("/approvals/fake_id/approve", json={})
        assert resp.status_code == 404

    def test_reject_nonexistent(self, client):
        resp = client.post("/approvals/fake_id/reject", json={})
        assert resp.status_code == 404

    def test_status_nonexistent(self, client):
        resp = client.get("/approvals/fake_id/status")
        assert resp.status_code == 404

    def test_approve_already_approved(self, client):
        gate = get_approval_gate()
        rid = gate.request_approval("test", cost_usdc=5.0)
        gate.approve(rid)
        resp = client.post(f"/approvals/{rid}/approve", json={})
        assert resp.status_code == 400

    def test_get_all_approvals(self, client):
        gate = get_approval_gate()
        gate.request_approval("a", cost_usdc=5.0)
        gate.auto_approve("b", cost_usdc=0.10)
        resp = client.get("/approvals/all")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_stats_after_operations(self, client):
        gate = get_approval_gate()
        r1 = gate.request_approval("a", cost_usdc=5.0)
        gate.approve(r1)
        r2 = gate.request_approval("b", cost_usdc=5.0)
        gate.reject(r2)
        gate.auto_approve("c")

        resp = client.get("/approvals/stats")
        data = resp.json()
        assert data["approved"] == 1
        assert data["rejected"] == 1
        assert data["auto_approved"] == 1

    def test_approve_with_default_body(self, client):
        gate = get_approval_gate()
        rid = gate.request_approval("test", cost_usdc=5.0)
        resp = client.post(f"/approvals/{rid}/approve")
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"
