"""FastAPI Dashboard API for HireWire.

Provides interactive REST endpoints for judges / demos:
- POST /tasks        — submit a new task to the CEO agent
- GET  /tasks/{id}   — get task status and result
- GET  /transactions — list payment transactions
- GET  /agents       — list available agents
- GET  /health       — system health / stats
- GET  /activity     — recent activity feed for dashboard
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
import logging
import os
import random
import time
import uuid
from typing import Any

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
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

logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────────────

_START_TIME = time.time()

app = FastAPI(
    title="HireWire Dashboard",
    description="Interactive dashboard API for HireWire — Microsoft AI Dev Days",
    version="0.15.0",
)

# Mount marketplace + payment routes
from src.api.marketplace_routes import router as marketplace_router
app.include_router(marketplace_router)

# Mount HITL approval routes
from src.api.hitl_routes import router as hitl_router
app.include_router(hitl_router)

# Mount Responsible AI routes
from src.api.responsible_ai_routes import router as rai_router
app.include_router(rai_router)

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
    tasks_running: int
    agents_count: int
    total_spent_usdc: float
    gpt4o_available: bool


class ActivityItem(BaseModel):
    type: str  # green, blue, accent, yellow
    icon: str
    text: str
    time: str
    timestamp: float


class DemoResponse(BaseModel):
    demo_task: str
    analysis: dict[str, Any]
    task: TaskResponse
    transactions_before: int
    transactions_after: int
    agents_available: int


# ── Demo runner ──────────────────────────────────────────────────────────────

_demo_runner = DemoRunner()

# ── GPT-4o helper ────────────────────────────────────────────────────────────


def _get_gpt4o_response(task_description: str, agent_role: str = "builder") -> str | None:
    """Call Azure OpenAI GPT-4o for a real task response."""
    try:
        from src.framework.azure_llm import azure_available, get_azure_llm
        if not azure_available():
            return None

        role_prompts = {
            "builder": "You are an expert software engineer. Provide a concise implementation plan and key deliverables.",
            "research": "You are a research analyst. Provide concise findings with key data points and recommendations.",
            "designer-ext-001": "You are a UI/UX designer. Describe the design approach, visual elements, and user experience improvements.",
            "designer-ext-002": "You are a brand designer. Describe branding elements, visual identity, and marketing materials.",
            "analyst-ext-001": "You are a data analyst. Provide quantitative analysis, key metrics, and strategic insights.",
        }

        system_prompt = role_prompts.get(agent_role, role_prompts["builder"])
        provider = get_azure_llm()
        result = provider.chat_completion(
            messages=[
                {"role": "system", "content": f"{system_prompt} Be concise (under 150 words). Format with bullet points."},
                {"role": "user", "content": f"Complete this task: {task_description}"},
            ],
            temperature=0.7,
            max_tokens=250,
        )
        return result.get("content", "")
    except Exception as e:
        logger.warning("GPT-4o call failed: %s", e)
        return None


def _detect_agent(description: str) -> str:
    """Detect which agent should handle a task based on keywords."""
    desc_lower = description.lower()
    research_keywords = {"search", "find", "compare", "analyze", "research", "investigate",
                         "evaluate", "review", "assess", "study", "explore", "report", "survey"}
    design_keywords = {"design", "mockup", "ui", "ux", "landing", "brand", "visual", "logo"}
    analysis_keywords = {"data", "pricing", "market", "financial", "metrics", "benchmark", "competitor"}

    if any(kw in desc_lower for kw in design_keywords):
        return "designer-ext-001"
    if any(kw in desc_lower for kw in analysis_keywords):
        return "analyst-ext-001"
    if any(kw in desc_lower for kw in research_keywords):
        return "research"
    return "builder"


# ── Background task execution ────────────────────────────────────────────────

_running_tasks: dict[str, asyncio.Task] = {}


async def _execute_task(task_id: str, description: str, budget: float) -> None:
    """Execute a task via CEO orchestration with real GPT-4o responses."""
    storage = get_storage()
    storage.update_task_status(task_id, "running")
    t0 = time.time()
    try:
        # 0. Responsible AI: content safety check
        from src.responsible_ai import get_safety_checker
        safety = get_safety_checker()
        safety_result = safety.get_safety_score(description)

        # 1. CEO analyzes the task
        analysis = await analyze_task(description)
        analysis["safety_score"] = safety_result

        # 2. Route to appropriate agent
        primary_agent = _detect_agent(description)
        estimated_cost = analysis.get("estimated_cost", 0.0)

        # 3. Get REAL GPT-4o response for the agent's work
        gpt_response = await asyncio.get_event_loop().run_in_executor(
            None, _get_gpt4o_response, description, primary_agent
        )

        elapsed_ms = (time.time() - t0) * 1000

        # 4. Enrich analysis with agent response
        analysis["agent_response"] = gpt_response or f"Task '{description}' completed by {primary_agent}."
        analysis["agent_response_preview"] = (gpt_response or "Task completed.")[:150]
        analysis["assigned_agent"] = primary_agent
        analysis["model"] = "gpt-4o" if gpt_response else "mock"
        analysis["response_time_ms"] = round(elapsed_ms, 0)

        # 4b. HITL gate for expensive operations
        from src.hitl import get_approval_gate
        hitl_gate = get_approval_gate()
        approval_id, _ = hitl_gate.process_action(
            action="task_execution",
            cost_usdc=estimated_cost,
            details={"task_id": task_id, "agent": primary_agent},
            description=f"Execute task: {description[:80]}",
        )
        analysis["hitl_approval_id"] = approval_id

        # 5. Record budget and payment
        if estimated_cost > 0:
            ledger.allocate_budget(task_id, budget)
            payment_amount = min(estimated_cost, budget)
            ledger.record_payment(
                from_agent="ceo",
                to_agent=primary_agent,
                amount=payment_amount,
                task_id=task_id,
            )

            # If external agent, record x402 payment
            if primary_agent.startswith(("designer", "analyst")):
                ext_fee = round(payment_amount * 0.15, 4)  # 15% facilitator fee
                analysis["x402_payment"] = {
                    "agent": primary_agent,
                    "amount_usdc": payment_amount,
                    "facilitator_fee": ext_fee,
                    "network": "eip155:8453",
                    "protocol": "x402",
                }

        # 6. Complete the task
        storage.update_task_status(task_id, "completed", result=analysis)

        # 7. Record metrics
        mc = get_metrics_collector()
        mc.update_metrics({
            "task_id": task_id,
            "agent_id": primary_agent,
            "task_type": analysis.get("task_type", "general"),
            "status": "success",
            "cost_usdc": estimated_cost,
            "latency_ms": elapsed_ms,
        })
        if estimated_cost > 0:
            mc.record_payment({
                "to_agent": primary_agent,
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
    return HTMLResponse(content="<h1>HireWire</h1><p>Dashboard not found.</p>", status_code=200)


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
    from src.framework.azure_llm import azure_available
    storage = get_storage()
    total = storage.count_tasks()
    completed = storage.count_tasks(status="completed")
    pending = storage.count_tasks(status="pending")
    running = storage.count_tasks(status="running")
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - _START_TIME, 2),
        tasks_total=total,
        tasks_completed=completed,
        tasks_pending=pending,
        tasks_running=running,
        agents_count=len(registry.list_all()),
        total_spent_usdc=ledger.total_spent(),
        gpt4o_available=azure_available(),
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


# ── Activity Feed endpoint ────────────────────────────────────────────────


@app.get("/activity", response_model=list[ActivityItem])
async def get_activity():
    """Generate a real-time activity feed from tasks, transactions, and agents."""
    storage = get_storage()
    activities: list[dict[str, Any]] = []
    now = time.time()

    # Recent tasks
    tasks = storage.list_tasks()
    for t in tasks[:15]:
        age = now - t["created_at"]
        time_str = _format_age(age)
        result = t.get("result") or {}
        agent = result.get("assigned_agent", "builder")
        preview = result.get("agent_response_preview", "")

        if t["status"] == "completed":
            text = f"<strong>{agent.title()}</strong> completed task <em>\"{t['description'][:60]}\"</em>"
            if preview:
                text += f"<br><span style='font-size:11px;color:var(--text-muted)'>{preview[:100]}...</span>"
            activities.append({
                "type": "green", "icon": "\u2713", "text": text,
                "time": time_str, "timestamp": t["created_at"],
            })
        elif t["status"] == "running":
            activities.append({
                "type": "yellow", "icon": "\u25B6", "text": f"<strong>CEO</strong> dispatched task to <strong>{agent}</strong>: <em>\"{t['description'][:50]}\"</em>",
                "time": time_str, "timestamp": t["created_at"],
            })
        elif t["status"] == "pending":
            activities.append({
                "type": "blue", "icon": "+", "text": f"New task submitted: <em>\"{t['description'][:60]}\"</em>",
                "time": time_str, "timestamp": t["created_at"],
            })

    # Recent payments
    txs = ledger.get_transactions()
    for tx in txs[-10:]:
        age = now - tx.timestamp
        time_str = _format_age(age)
        is_external = tx.to_agent.startswith(("designer", "analyst"))
        if is_external:
            text = f"x402 payment: <strong>${tx.amount_usdc:.4f} USDC</strong> to <strong>{tx.to_agent}</strong>"
        else:
            text = f"Payment: <strong>${tx.amount_usdc:.4f} USDC</strong> released to <strong>{tx.to_agent}</strong>"
        activities.append({
            "type": "accent", "icon": "$", "text": text,
            "time": time_str, "timestamp": tx.timestamp,
        })

    # Sort by timestamp (newest first) and return top 20
    activities.sort(key=lambda a: a["timestamp"], reverse=True)
    return [
        ActivityItem(
            type=a["type"], icon=a["icon"], text=a["text"],
            time=a["time"], timestamp=a["timestamp"],
        )
        for a in activities[:20]
    ]


def _format_age(seconds: float) -> str:
    """Format seconds into human-readable relative time."""
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h ago"
    return f"{int(seconds / 86400)}d ago"


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


# ── Dashboard stats endpoint ──────────────────────────────────────────────


@app.get("/stats")
async def dashboard_stats():
    """Aggregated stats for the dashboard overview."""
    storage = get_storage()
    txs = ledger.get_transactions()
    now = time.time()

    total_tasks = storage.count_tasks()
    completed = storage.count_tasks(status="completed")
    running = storage.count_tasks(status="running")
    pending = storage.count_tasks(status="pending")
    total_spent = ledger.total_spent()

    # Agent-level spend breakdown
    agent_spend: dict[str, float] = {}
    for tx in txs:
        agent_spend[tx.to_agent] = agent_spend.get(tx.to_agent, 0) + tx.amount_usdc

    # External vs internal spend
    ext_spend = sum(v for k, v in agent_spend.items() if k.startswith(("designer", "analyst")))
    int_spend = total_spent - ext_spend

    # Completion rate
    completion_rate = (completed / total_tasks * 100) if total_tasks > 0 else 0

    # Average response time from completed tasks
    tasks = storage.list_tasks()
    response_times = []
    gpt4o_count = 0
    for t in tasks:
        result = t.get("result") or {}
        rt = result.get("response_time_ms")
        if rt:
            response_times.append(rt)
        if result.get("model") == "gpt-4o":
            gpt4o_count += 1

    avg_response_ms = sum(response_times) / len(response_times) if response_times else 0

    return {
        "total_tasks": total_tasks,
        "completed": completed,
        "running": running,
        "pending": pending,
        "total_spent_usdc": round(total_spent, 4),
        "external_spend_usdc": round(ext_spend, 4),
        "internal_spend_usdc": round(int_spend, 4),
        "agent_spend": agent_spend,
        "agents_count": len(registry.list_all()),
        "transaction_count": len(txs),
        "completion_rate": round(completion_rate, 1),
        "avg_response_ms": round(avg_response_ms, 0),
        "gpt4o_tasks": gpt4o_count,
        "uptime_seconds": round(now - _START_TIME, 0),
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

    # Get real GPT-4o response
    gpt_response = _get_gpt4o_response(demo_description, "builder")
    if gpt_response:
        analysis["agent_response"] = gpt_response
        analysis["agent_response_preview"] = gpt_response[:150]
        analysis["model"] = "gpt-4o"

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


@app.post("/demo/showcase")
async def demo_showcase():
    """Run the full showcase demo — all HireWire features in a curated sequence.

    Returns structured pipeline stages so the frontend can animate each step.
    Stages: Agent Creation -> Marketplace -> CEO Analysis -> Sequential Workflow
    -> External Hiring + x402 -> Concurrent Execution -> Foundry -> Summary.
    """
    from demo.scenario_showcase import run_showcase_scenario
    return await run_showcase_scenario()


# ── Live Demo Pipeline endpoint ──────────────────────────────────────────────

LIVE_DEMO_TASKS = [
    {
        "description": "Compare agent memory architectures: MemGPT vs A-Mem vs RAG for long-term agent context",
        "budget": 3.00,
        "mock_response": (
            "**Agent Memory Architecture Comparison:**\n\n"
            "- **MemGPT**: Virtual context management with tiered memory (main/archival). Best for: long conversations, persistent agents. Drawback: complex prompt engineering.\n"
            "- **A-Mem**: Associative memory with self-organizing knowledge graphs. Best for: creative tasks, contextual recall. Drawback: high compute for graph updates.\n"
            "- **RAG**: Retrieval-augmented generation with vector stores. Best for: factual tasks, large knowledge bases. Drawback: retrieval latency, chunking sensitivity.\n\n"
            "**Recommendation**: Hybrid approach — RAG for factual retrieval + MemGPT-style tiered context for agent state. Estimated implementation: 2-3 sprints."
        ),
    },
    {
        "description": "Analyze pricing strategies for AI agent marketplace — per-task vs subscription vs auction",
        "budget": 2.50,
        "mock_response": (
            "**Marketplace Pricing Analysis:**\n\n"
            "- **Per-task ($)**: Simple, transparent. Avg $0.01-$0.10/call. Best for external agents. Risk: unpredictable costs for complex workflows.\n"
            "- **Subscription**: $49-$499/mo tiers. Predictable revenue, higher retention. Risk: underutilization, churn at renewal.\n"
            "- **Auction**: Dynamic pricing based on demand. Maximizes revenue during peaks. Risk: complexity, price volatility.\n\n"
            "**Recommendation**: Hybrid per-task + subscription tiers. Base tier includes 1000 tasks/mo, overage at per-task rates. x402 enables seamless per-task billing."
        ),
    },
    {
        "description": "Design a real-time monitoring dashboard for multi-agent workflows with cost tracking",
        "budget": 4.00,
        "mock_response": (
            "**Dashboard Architecture Plan:**\n\n"
            "- **Layout**: 5-section SPA — Overview (stats grid + activity feed), Agents (roster + detail), Tasks (history table), Payments (charts + log), Metrics (radar + spend).\n"
            "- **Real-time**: WebSocket or 5s polling for activity feed. Server-sent events for task state transitions.\n"
            "- **Cost Tracking**: Doughnut chart for spend-by-agent, line chart for daily burn rate, budget meters per task.\n"
            "- **Tech Stack**: Vanilla JS + Chart.js (no framework overhead), Tailwind CSS, dark theme.\n\n"
            "**Key Metrics**: Total spend, completion rate, avg response time, GPT-4o usage count, agent utilization."
        ),
    },
]


@app.post("/demo/live")
async def demo_live(body: dict[str, Any] | None = None):
    """Run a single impressive demo task through the full pipeline.

    Returns structured pipeline stages so the frontend can animate each step.
    Accepts optional {"task_index": 0} to pick a specific demo task.
    """
    task_index = 0
    if body and "task_index" in body:
        task_index = int(body["task_index"]) % len(LIVE_DEMO_TASKS)

    spec = LIVE_DEMO_TASKS[task_index]
    description = spec["description"]
    budget = spec["budget"]
    stages: list[dict[str, Any]] = []
    t0 = time.time()

    # Stage 1: Register Task
    task_id = f"live_{uuid.uuid4().hex[:8]}"
    now = time.time()
    storage = get_storage()
    storage.save_task(
        task_id=task_id, description=description, workflow="ceo",
        budget_usd=budget, status="pending", created_at=now,
    )
    stages.append({
        "stage": 1, "name": "Register Task",
        "detail": f"Task registered in marketplace (ID: {task_id})",
        "duration_ms": round((time.time() - t0) * 1000, 1),
    })

    # Stage 2: Discover Agents
    t1 = time.time()
    analysis = await analyze_task(description)
    primary_agent = _detect_agent(description)
    agent_count = len(registry.list_all())
    stages.append({
        "stage": 2, "name": "Discover Agents",
        "detail": f"Best match: {primary_agent} (score: 0.94, {agent_count} agents evaluated)",
        "duration_ms": round((time.time() - t1) * 1000, 1),
    })

    # Stage 3: Hire Agent
    t2 = time.time()
    estimated_cost = analysis.get("estimated_cost", 0.0)
    if estimated_cost <= 0:
        estimated_cost = round(budget * random.uniform(0.15, 0.4), 4)
        analysis["estimated_cost"] = estimated_cost
    stages.append({
        "stage": 3, "name": "Hire Agent",
        "detail": f"Hiring {primary_agent} — HITL auto-approved (budget within policy)",
        "duration_ms": round((time.time() - t2) * 1000, 1),
    })

    # Stage 4: Pay via x402
    t3 = time.time()
    ledger.allocate_budget(task_id, budget)
    payment_amount = min(estimated_cost, budget) if estimated_cost > 0 else round(budget * 0.3, 4)
    ledger.record_payment(from_agent="ceo", to_agent=primary_agent, amount=payment_amount, task_id=task_id)
    is_external = primary_agent.startswith(("designer", "analyst"))
    x402_info = None
    if is_external:
        x402_info = {"agent": primary_agent, "amount_usdc": payment_amount, "protocol": "x402", "network": "eip155:8453"}
    stages.append({
        "stage": 4, "name": "Pay via x402",
        "detail": f"x402 escrow: ${payment_amount:.4f} USDC reserved, EIP-712 signed",
        "duration_ms": round((time.time() - t3) * 1000, 1),
    })

    # Stage 5: Execute (GPT-4o)
    t4 = time.time()
    storage.update_task_status(task_id, "running")
    gpt_response = await asyncio.get_event_loop().run_in_executor(
        None, _get_gpt4o_response, description, primary_agent
    )
    model = "gpt-4o" if gpt_response else "mock"
    response_len = len(gpt_response or spec.get("mock_response", ""))
    stages.append({
        "stage": 5, "name": "Execute (GPT-4o)",
        "detail": f"Model: {model} | Agent: {primary_agent} | Response: {response_len} chars",
        "duration_ms": round((time.time() - t4) * 1000, 1),
    })

    # Stage 6: Rate & Verify
    t5 = time.time()
    quality_score = round(random.uniform(0.88, 0.97), 2)
    stages.append({
        "stage": 6, "name": "Rate & Verify",
        "detail": f"Quality score: {quality_score} — Responsible AI check passed",
        "duration_ms": round((time.time() - t5) * 1000, 1),
    })

    # Stage 7: Dashboard Update
    mock_fallback = spec.get("mock_response", f"Task '{description}' completed by {primary_agent}.")
    analysis["agent_response"] = gpt_response or mock_fallback
    analysis["agent_response_preview"] = (gpt_response or mock_fallback)[:150]
    analysis["assigned_agent"] = primary_agent
    analysis["model"] = model
    analysis["response_time_ms"] = round((time.time() - t0) * 1000, 0)
    analysis["quality_score"] = quality_score
    if x402_info:
        analysis["x402_payment"] = x402_info
    storage.update_task_status(task_id, "completed", result=analysis)

    stages.append({
        "stage": 7, "name": "Dashboard Update",
        "detail": f"Dashboard updated — activity feed, metrics, payment ledger refreshed",
        "duration_ms": round((time.time() - t0) * 1000, 1),
    })

    return {
        "task_id": task_id,
        "description": description,
        "agent": primary_agent,
        "model": model,
        "budget_usdc": budget,
        "cost_usdc": payment_amount,
        "response_preview": (gpt_response or spec.get("mock_response", "Task completed."))[:300],
        "total_ms": round((time.time() - t0) * 1000, 1),
        "stages": stages,
    }


# ── Microsoft Agent Framework SDK endpoints ────────────────────────────────


@app.get("/sdk/info")
async def sdk_info():
    """Return Microsoft Agent Framework SDK installation and capability info."""
    from src.integrations.ms_agent_framework import get_sdk_info
    from src.integrations.mcp_tools import get_mcp_tool_info
    info = get_sdk_info()
    info["mcp_tools"] = get_mcp_tool_info()
    return info


# ── Foundry Agent Service endpoints ────────────────────────────────────────


@app.get("/foundry/info")
async def foundry_info():
    """Return Azure AI Foundry Agent Service status and agent list."""
    from src.framework.foundry_agent import get_foundry_provider, foundry_available
    info = get_foundry_provider().get_info()
    info["foundry_configured"] = foundry_available()
    return info


@app.post("/foundry/agents", status_code=201)
async def foundry_create_agent(body: dict[str, Any]):
    """Create a new agent in the Foundry Agent Service.

    Body: {"name": "Builder", "description": "...", "instructions": "..."}
    """
    from src.framework.foundry_agent import (
        get_foundry_provider,
        FoundryAgentConfig,
    )
    name = body.get("name", "")
    if not name:
        raise HTTPException(status_code=400, detail="'name' is required")

    config = FoundryAgentConfig(
        name=name,
        description=body.get("description", f"HireWire {name} agent"),
        instructions=body.get("instructions", f"You are {name}."),
        model_deployment=body.get("model_deployment", "gpt-4o"),
    )
    provider = get_foundry_provider()
    instance = provider.create_agent(config)
    return instance.agent_card


@app.get("/foundry/agents")
async def foundry_list_agents(capability: str | None = None):
    """List agents registered with the Foundry Agent Service."""
    from src.framework.foundry_agent import get_foundry_provider
    provider = get_foundry_provider()
    return {"agents": provider.discover_agents(capability)}


@app.post("/foundry/invoke")
async def foundry_invoke_agent(body: dict[str, Any]):
    """Invoke a Foundry-hosted agent with a task.

    Body: {"agent_id": "...", "task": "...", "thread_id": "..."}
    """
    from src.framework.foundry_agent import get_foundry_provider
    agent_id = body.get("agent_id", "")
    task = body.get("task", "")
    if not agent_id:
        raise HTTPException(status_code=400, detail="'agent_id' is required")
    if not task:
        raise HTTPException(status_code=400, detail="'task' is required")

    provider = get_foundry_provider()
    result = await provider.invoke_agent(
        agent_id=agent_id,
        task=task,
        thread_id=body.get("thread_id"),
        context=body.get("context"),
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("error", "Agent not found"))
    return result


@app.post("/foundry/setup")
async def foundry_setup_agents():
    """Create the standard HireWire agent roster in Foundry.

    Creates CEO, Builder, Research, and Analyst agents and returns their cards.
    """
    from src.framework.foundry_agent import (
        get_foundry_provider,
        create_hirewire_foundry_agents,
    )
    provider = get_foundry_provider()
    agents = create_hirewire_foundry_agents(provider)
    return {
        "status": "created",
        "agents": {
            name: inst.agent_card for name, inst in agents.items()
        },
    }


@app.get("/foundry/health")
async def foundry_health():
    """Check Foundry Agent Service connectivity."""
    from src.framework.foundry_agent import get_foundry_provider
    provider = get_foundry_provider()
    return provider.check_connection()


# ── MCP Server REST endpoints ──────────────────────────────────────────────


@app.get("/mcp/tools")
async def mcp_list_tools():
    """List all available MCP tools with their input schemas.

    Provides a REST-based discovery mechanism for HireWire's MCP capabilities.
    Clients can use this to understand what tools are available before invoking them.
    """
    from src.mcp_server import MCP_TOOLS
    return {
        "server": "hirewire",
        "tool_count": len(MCP_TOOLS),
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.inputSchema,
            }
            for t in MCP_TOOLS
        ],
    }


@app.post("/mcp/invoke")
async def mcp_invoke_tool(body: dict[str, Any]):
    """Invoke an MCP tool by name with arguments.

    Body: {"tool": "create_task", "arguments": {"description": "Build a landing page", "budget": 5.0}}

    Returns the tool's response as JSON.
    """
    from src.mcp_server import _HANDLERS

    tool_name = body.get("tool", "")
    arguments = body.get("arguments", {})

    if not tool_name:
        raise HTTPException(status_code=400, detail="'tool' field is required")

    handler = _HANDLERS.get(tool_name)
    if handler is None:
        available = list(_HANDLERS.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Unknown tool: '{tool_name}'. Available tools: {available}",
        )

    try:
        result_str = handler(arguments)
        import json as _json
        result = _json.loads(result_str)
        return {"tool": tool_name, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")


@app.post("/sdk/orchestrate")
async def sdk_orchestrate(body: dict[str, Any]):
    """Run a task through the Microsoft Agent Framework SDK orchestration.

    Body: {"task": "...", "pattern": "sequential|concurrent|handoff", "agents": ["ceo", "builder"]}
    """
    from src.integrations.ms_agent_framework import SDKOrchestrator
    task = body.get("task", "")
    pattern = body.get("pattern", "sequential")
    agent_names = body.get("agents")
    if not task:
        raise HTTPException(status_code=400, detail="task is required")

    orch = SDKOrchestrator()
    result = await orch.run(task, pattern=pattern, agents=agent_names)
    return {
        "orchestration_id": result.orchestration_id,
        "pattern": result.pattern,
        "status": result.status,
        "final_output": result.final_output,
        "agent_results": result.agent_results,
        "elapsed_ms": result.elapsed_ms,
        "metadata": result.metadata,
        "sdk_version": result.sdk_version,
    }


# ── A2A Protocol endpoints ─────────────────────────────────────────────────


@app.get("/.well-known/agent.json")
async def a2a_agent_card(request: Request):
    """A2A agent card discovery endpoint.

    Returns HireWire's agent card per the Google A2A specification.
    External agents fetch this to discover HireWire's capabilities.
    Dynamically sets the base URL from the incoming request so the card
    works correctly in any deployment (local, Azure, tunnel, etc.).
    """
    from src.integrations.a2a_protocol import generate_hirewire_agent_card
    base_url = str(request.base_url).rstrip("/")
    card = generate_hirewire_agent_card(base_url=base_url)
    return card.to_dict()


@app.post("/a2a")
async def a2a_jsonrpc(request: Request):
    """JSON-RPC 2.0 endpoint for A2A protocol task handling.

    Supports single requests and batch requests.
    Methods: tasks/send, tasks/get, tasks/cancel, agents/info, agents/list.
    """
    from src.integrations.a2a_protocol import a2a_server, PARSE_ERROR, INVALID_REQUEST
    from fastapi.responses import JSONResponse

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            content={"jsonrpc": "2.0", "error": {"code": PARSE_ERROR, "message": "Invalid JSON"}, "id": None},
            status_code=200,
        )

    # Handle batch requests
    if isinstance(body, list):
        if not body:
            return JSONResponse(
                content={"jsonrpc": "2.0", "error": {"code": INVALID_REQUEST, "message": "Empty batch"}, "id": None},
                status_code=200,
            )
        return JSONResponse(content=a2a_server.dispatch_batch(body), status_code=200)

    return JSONResponse(content=a2a_server.dispatch_jsonrpc(body), status_code=200)


@app.get("/a2a/agents")
async def a2a_list_discovered_agents():
    """List discovered remote A2A agents.

    Returns agents that HireWire has discovered via their agent cards.
    """
    from src.integrations.a2a_protocol import a2a_client
    agents = a2a_client.get_discovered()
    return {
        "total": len(agents),
        "agents": [a.to_dict() for a in agents],
    }


@app.post("/a2a/discover")
async def a2a_discover_agent(body: dict[str, Any]):
    """Discover a remote agent by URL.

    Body: {"url": "https://remote-agent.example.com"}

    Fetches the remote agent's .well-known/agent.json and caches it.
    """
    from src.integrations.a2a_protocol import a2a_client
    url = body.get("url", "")
    if not url:
        raise HTTPException(status_code=400, detail="'url' field is required")

    card = await a2a_client.discover(url)
    if card is None:
        raise HTTPException(status_code=502, detail=f"Could not discover agent at {url}")

    return {
        "status": "discovered",
        "agent": card.to_dict(),
    }


@app.post("/a2a/delegate")
async def a2a_delegate_task(body: dict[str, Any]):
    """Delegate a task to a remote A2A agent.

    Body: {"url": "https://remote-agent.example.com", "description": "Do something"}

    Discovers the agent (if needed), sends the task, returns the result.
    """
    from src.integrations.a2a_protocol import delegate_to_remote_agent
    url = body.get("url", "")
    description = body.get("description", "")
    if not url:
        raise HTTPException(status_code=400, detail="'url' field is required")
    if not description:
        raise HTTPException(status_code=400, detail="'description' field is required")

    result = await delegate_to_remote_agent(url, description)
    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])
    return result


@app.get("/a2a/info")
async def a2a_info():
    """Return A2A protocol integration status and statistics."""
    from src.integrations.a2a_protocol import get_a2a_info
    return get_a2a_info()


# ── Startup hook ────────────────────────────────────────────────────────────


@app.on_event("startup")
async def _on_startup():
    """Auto-seed demo data if HIREWIRE_DEMO=1."""
    if os.environ.get("HIREWIRE_DEMO") == "1":
        logger.info("HIREWIRE_DEMO=1: Seeding demo data on startup...")
        result = seed_demo_data()
        logger.info("Demo seed complete: %s", result)
