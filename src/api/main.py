"""FastAPI Dashboard API for AgentOS.

Provides interactive REST endpoints for judges / demos:
- POST /tasks        — submit a new task to the CEO agent
- GET  /tasks/{id}   — get task status and result
- GET  /transactions — list payment transactions
- GET  /agents       — list available agents
- GET  /health       — system health / stats
- GET  /demo         — run a pre-configured demo scenario

Start standalone:
    uvicorn src.api.main:app --port 8000
"""

from __future__ import annotations

import asyncio
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
from src.mcp_servers.payment_hub import ledger, PaymentRecord
from src.mcp_servers.registry_server import registry
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


# ── Background task execution ────────────────────────────────────────────────

_running_tasks: dict[str, asyncio.Task] = {}


async def _execute_task(task_id: str, description: str, budget: float) -> None:
    """Analyse the task via CEO tools and persist the result."""
    storage = get_storage()
    storage.update_task_status(task_id, "running")
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
    except Exception as exc:
        storage.update_task_status(task_id, "failed", result={"error": str(exc)})
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
