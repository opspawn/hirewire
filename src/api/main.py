"""FastAPI Dashboard API for AgentOS.

Provides interactive REST endpoints for judges / demos:
- POST /tasks        — submit a new task to the CEO agent
- GET  /tasks/{id}   — get task status and result
- GET  /transactions — list payment transactions
- GET  /agents       — list available agents
- GET  /health       — system health / stats
- GET  /demo         — run a pre-configured demo scenario
- GET  /demo/start   — start live demo runner
- GET  /demo/stop    — stop live demo runner
- GET  /demo/seed    — seed demo data
- GET  /demo/status  — get demo runner status

Start standalone:
    uvicorn src.api.main:app --port 8000
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import asdict
from typing import Any

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agents.ceo_agent import analyze_task
from src.demo.runner import DemoRunner
from src.demo.seeder import seed_demo_data
from src.mcp_servers.payment_hub import ledger, PaymentRecord
from src.mcp_servers.registry_server import registry
from src.metrics.collector import get_metrics_collector
from src.metrics.analytics import CostAnalyzer, ROICalculator
from src.storage import get_storage

# ── App ──────────────────────────────────────────────────────────────────────

_START_TIME = time.time()

app = FastAPI(
    title="AgentOS Dashboard",
    description="Interactive dashboard API for AgentOS — Microsoft AI Dev Days",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Dashboard static files ──────────────────────────────────────────────────

_DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
if _DASHBOARD_DIR.is_dir():
    app.mount("/dashboard", StaticFiles(directory=str(_DASHBOARD_DIR), html=True), name="dashboard")

# ── Request / Response models ────────────────────────────────────────────────


class TaskSubmission(BaseModel):
    description: str = Field(..., min_length=1, max_length=2000)
    budget: float = Field(default=1.0, gt=0, le=1000)


class TaskResponse(BaseModel):
    task_id: str
    description: str
    status: str
    budget_usd: float
    created_at: float
    result: dict[str, Any] | None = None


class TransactionResponse(BaseModel):
    tx_id: str
    from_agent: str
    to_agent: str
    amount_usdc: float
    task_id: str
    timestamp: float
    status: str
    tx_hash: str


class AgentResponse(BaseModel):
    name: str
    description: str
    skills: list[str]
    price_per_call: str
    endpoint: str
    protocol: str
    payment: str
    is_external: bool


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    tasks_total: int
    tasks_completed: int
    tasks_pending: int
    agents_count: int
    total_spent_usdc: float


class DemoResponse(BaseModel):
    demo_task: str
    analysis: dict[str, Any]
    task: TaskResponse
    transactions_before: int
    transactions_after: int
    agents_available: int


# ── Demo runner ──────────────────────────────────────────────────────────────

_demo_runner = DemoRunner()

# ── Background task execution ────────────────────────────────────────────────

_running_tasks: dict[str, asyncio.Task] = {}


async def _execute_task(task_id: str, description: str, budget: float) -> None:
    """Analyse the task via CEO tools and persist the result."""
    storage = get_storage()
    storage.update_task_status(task_id, "running")
    t0 = time.time()
    try:
        analysis = await analyze_task(description)
        # Record a simulated payment for the estimated cost
        estimated_cost = analysis.get("estimated_cost", 0.0)
        if estimated_cost > 0:
            ledger.allocate_budget(task_id, budget)
            ledger.record_payment(
                from_agent="ceo",
                to_agent="builder",
                amount=min(estimated_cost, budget),
                task_id=task_id,
            )
        storage.update_task_status(task_id, "completed", result=analysis)
        # Record metrics
        elapsed_ms = (time.time() - t0) * 1000
        mc = get_metrics_collector()
        mc.update_metrics({
            "task_id": task_id,
            "agent_id": "ceo",
            "task_type": analysis.get("task_type", "general"),
            "status": "success",
            "cost_usdc": estimated_cost,
            "latency_ms": elapsed_ms,
        })
        if estimated_cost > 0:
            mc.record_payment({
                "to_agent": "builder",
                "task_id": task_id,
                "amount_usdc": min(estimated_cost, budget),
                "status": "completed",
            })
    except Exception as exc:
        storage.update_task_status(task_id, "failed", result={"error": str(exc)})
        elapsed_ms = (time.time() - t0) * 1000
        mc = get_metrics_collector()
        mc.update_metrics({
            "task_id": task_id,
            "agent_id": "ceo",
            "task_type": "general",
            "status": "failure",
            "cost_usdc": 0.0,
            "latency_ms": elapsed_ms,
        })
    finally:
        _running_tasks.pop(task_id, None)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard or redirect to /dashboard."""
    index = _DASHBOARD_DIR / "index.html"
    if index.is_file():
        return HTMLResponse(content=index.read_text(), status_code=200)
    return HTMLResponse(content="<h1>AgentOS</h1><p>Dashboard not found.</p>", status_code=200)


@app.get("/tasks", response_model=list[TaskResponse])
async def list_tasks():
    """List all tasks (most recent first)."""
    storage = get_storage()
    rows = storage.list_tasks()
    return [
        TaskResponse(
            task_id=r["task_id"],
            description=r["description"],
            status=r["status"],
            budget_usd=r["budget_usd"],
            created_at=r["created_at"],
            result=r.get("result"),
        )
        for r in rows
    ]


@app.post("/tasks", response_model=TaskResponse, status_code=201)
async def submit_task(body: TaskSubmission):
    """Submit a new task to the CEO agent."""
    task_id = f"task_{uuid.uuid4().hex[:12]}"
    now = time.time()
    storage = get_storage()
    storage.save_task(
        task_id=task_id,
        description=body.description,
        workflow="ceo",
        budget_usd=body.budget,
        status="pending",
        created_at=now,
    )
    bg = asyncio.create_task(_execute_task(task_id, body.description, body.budget))
    _running_tasks[task_id] = bg
    return TaskResponse(
        task_id=task_id,
        description=body.description,
        status="pending",
        budget_usd=body.budget,
        created_at=now,
    )


@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Get task status and result."""
    record = get_storage().get_task(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return TaskResponse(
        task_id=record["task_id"],
        description=record["description"],
        status=record["status"],
        budget_usd=record["budget_usd"],
        created_at=record["created_at"],
        result=record["result"],
    )


@app.get("/transactions", response_model=list[TransactionResponse])
async def list_transactions():
    """List all payment transactions."""
    txs = ledger.get_transactions()
    return [
        TransactionResponse(
            tx_id=t.tx_id,
            from_agent=t.from_agent,
            to_agent=t.to_agent,
            amount_usdc=t.amount_usdc,
            task_id=t.task_id,
            timestamp=t.timestamp,
            status=t.status,
            tx_hash=t.tx_hash,
        )
        for t in txs
    ]


@app.get("/agents", response_model=list[AgentResponse])
async def list_agents():
    """List available agents from the registry."""
    agents = registry.list_all()
    return [
        AgentResponse(
            name=a.name,
            description=a.description,
            skills=a.skills,
            price_per_call=a.price_per_call,
            endpoint=a.endpoint,
            protocol=a.protocol,
            payment=a.payment,
            is_external=a.is_external,
        )
        for a in agents
    ]


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with system stats."""
    storage = get_storage()
    total = storage.count_tasks()
    completed = storage.count_tasks(status="completed")
    pending = storage.count_tasks(status="pending")
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - _START_TIME, 2),
        tasks_total=total,
        tasks_completed=completed,
        tasks_pending=pending,
        agents_count=len(registry.list_all()),
        total_spent_usdc=ledger.total_spent(),
    )


@app.get("/health/azure")
async def health_azure():
    """Deep health check that verifies Azure service connectivity."""
    from src.framework.azure_llm import azure_available, get_azure_llm
    from src.persistence.cosmos import cosmos_available, get_cosmos_store

    checks: dict[str, Any] = {}

    # Azure OpenAI check
    if azure_available():
        try:
            provider = get_azure_llm()
            checks["azure_openai"] = provider.check_connection()
        except Exception as exc:
            checks["azure_openai"] = {"connected": False, "error": str(exc)}
    else:
        checks["azure_openai"] = {"connected": False, "error": "Not configured"}

    # Cosmos DB check
    if cosmos_available():
        try:
            store = get_cosmos_store()
            checks["cosmos_db"] = store.check_connection()
        except Exception as exc:
            checks["cosmos_db"] = {"connected": False, "error": str(exc)}
    else:
        checks["cosmos_db"] = {"connected": False, "error": "Not configured"}

    all_connected = all(c.get("connected", False) for c in checks.values())
    return {
        "status": "healthy" if all_connected else "degraded",
        "services": checks,
    }


# ── Metrics endpoints ──────────────────────────────────────────────────────


@app.get("/metrics")
async def system_metrics():
    """System-wide metrics summary."""
    mc = get_metrics_collector()
    return mc.get_system_metrics()


@app.get("/metrics/agents")
async def agent_metrics():
    """Per-agent metrics summaries."""
    mc = get_metrics_collector()
    return mc.get_all_agent_summaries()


@app.get("/metrics/costs")
async def cost_metrics():
    """Cost analytics: by agent, by task type, efficiency, trends."""
    storage = get_storage()
    analyzer = CostAnalyzer(storage)
    roi = ROICalculator(storage)
    return {
        "cost_by_agent": analyzer.cost_by_agent(),
        "cost_by_task_type": analyzer.cost_by_task_type(),
        "efficiency": analyzer.efficiency_score(),
        "trend": analyzer.trend_analysis(),
        "savings": roi.savings_estimate(),
        "best_value_agents": roi.best_value_agents(),
    }


# ── Demo endpoints ─────────────────────────────────────────────────────────


@app.get("/demo", response_model=DemoResponse)
async def run_demo():
    """Run a pre-configured demo scenario and return results."""
    demo_description = "Build a landing page for an AI startup with modern design"
    demo_budget = 5.0

    txs_before = len(ledger.get_transactions())
    agents_count = len(registry.list_all())

    # Analyze the task
    analysis = await analyze_task(demo_description)

    # Create and persist the task
    task_id = f"demo_{uuid.uuid4().hex[:8]}"
    now = time.time()
    storage = get_storage()
    storage.save_task(
        task_id=task_id,
        description=demo_description,
        workflow="ceo",
        budget_usd=demo_budget,
        status="pending",
        created_at=now,
    )

    # Simulate execution
    estimated_cost = analysis.get("estimated_cost", 0.0)
    if estimated_cost > 0:
        ledger.allocate_budget(task_id, demo_budget)
        ledger.record_payment(
            from_agent="ceo",
            to_agent="builder",
            amount=min(estimated_cost, demo_budget),
            task_id=task_id,
        )
    storage.update_task_status(task_id, "completed", result=analysis)

    txs_after = len(ledger.get_transactions())

    task_record = storage.get_task(task_id)
    return DemoResponse(
        demo_task=demo_description,
        analysis=analysis,
        task=TaskResponse(
            task_id=task_record["task_id"],
            description=task_record["description"],
            status=task_record["status"],
            budget_usd=task_record["budget_usd"],
            created_at=task_record["created_at"],
            result=task_record["result"],
        ),
        transactions_before=txs_before,
        transactions_after=txs_after,
        agents_available=agents_count,
    )


# ── Live Demo Mode endpoints ────────────────────────────────────────────────


@app.get("/demo/seed")
async def demo_seed():
    """Seed the database with realistic demo data."""
    result = seed_demo_data()
    return {"status": "seeded", **result}


@app.get("/demo/start")
async def demo_start():
    """Start the live demo runner (submits tasks every 30s)."""
    if _demo_runner.is_running:
        return {"status": "already_running", **_demo_runner.status()}
    _demo_runner.start()
    return {"status": "started", **_demo_runner.status()}


@app.get("/demo/stop")
async def demo_stop():
    """Stop the live demo runner."""
    was_running = _demo_runner.is_running
    _demo_runner.stop()
    return {"status": "stopped", "was_running": was_running, **_demo_runner.status()}


@app.get("/demo/status")
async def demo_status():
    """Get current demo runner status."""
    return _demo_runner.status()


# ── Startup hook ────────────────────────────────────────────────────────────


@app.on_event("startup")
async def _on_startup():
    """Auto-seed demo data if AGENTOS_DEMO=1."""
    if os.environ.get("AGENTOS_DEMO") == "1":
        seed_demo_data()
