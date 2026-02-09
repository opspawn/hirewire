"""Demo Scenario: Agent Hiring — CEO discovers, hires, and pays an external designer.

Full flow:
1. CEO receives task: "Build a landing page with a professional design"
2. Internal Builder agent writes the code skeleton
3. CEO discovers an external designer agent in the marketplace
4. CEO hires the designer, sends task via A2A
5. Designer returns design specs, CEO aggregates results
6. Payment recorded via x402 ledger

Works with MODEL_PROVIDER=mock (default) or any real provider.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx
import uvicorn

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import get_chat_client, get_settings
from src.external.mock_agent import create_mock_agent_app, clear_processed_tasks
from src.mcp_servers.payment_hub import ledger
from src.mcp_servers.registry_server import registry
from src.workflows.hiring import run_hiring_workflow
from src.workflows.sequential import create_sequential_workflow, _extract_output_text


TASK_DESCRIPTION = (
    "Build a landing page with a professional design for AgentOS — "
    "an AI agent operating system that lets agents hire and pay each other"
)

TASK_ID = "demo-agent-hiring"
BUDGET_USD = 5.0
DESIGNER_PORT = 9100

OUTPUT_DIR = Path(__file__).parent / "output"


# -- ANSI helpers --

class _C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


def _header(text: str) -> None:
    print(f"\n{_C.BOLD}{_C.CYAN}{'=' * 60}{_C.RESET}")
    print(f"{_C.BOLD}{_C.CYAN}  {text}{_C.RESET}")
    print(f"{_C.BOLD}{_C.CYAN}{'=' * 60}{_C.RESET}\n")


def _step(num: int, text: str) -> None:
    print(f"{_C.BOLD}{_C.YELLOW}[Step {num}]{_C.RESET} {text}")


def _agent(name: str, action: str) -> None:
    print(f"  {_C.MAGENTA}{name}{_C.RESET} -> {action}")


def _ok(text: str) -> None:
    print(f"  {_C.GREEN}\u2713{_C.RESET} {text}")


def _info(text: str) -> None:
    print(f"  {_C.DIM}{text}{_C.RESET}")


def _money(text: str) -> None:
    print(f"  {_C.BLUE}${_C.RESET} {text}")


@contextlib.asynccontextmanager
async def _run_designer_server(port: int = DESIGNER_PORT):
    """Start the mock designer agent in the background."""
    app = create_mock_agent_app(port=port)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # Wait until the server is ready
    for _ in range(50):
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"http://127.0.0.1:{port}/health")
                if r.status_code == 200:
                    break
        except httpx.ConnectError:
            pass
        await asyncio.sleep(0.1)
    try:
        yield server
    finally:
        server.should_exit = True
        await task


async def run_agent_hiring_scenario() -> dict:
    """Run the full agent hiring demo scenario.

    Returns:
        Dict with keys: task, status, internal_result, hiring_result,
        budget, elapsed_s
    """
    _header("AgentOS Demo: External Agent Hiring")

    provider = get_settings().model_provider.value
    _info(f"Model provider: {provider}")
    _info(f"Task: {TASK_DESCRIPTION}")
    print()

    t0 = time.monotonic()
    clear_processed_tasks()

    # 1 — Budget allocation
    _step(1, "Allocating budget")
    ledger.allocate_budget(TASK_ID, BUDGET_USD)
    budget = ledger.get_budget(TASK_ID)
    _ok(f"Budget: ${budget.allocated:.2f} USDC allocated")
    print()

    # 2 — Internal builder writes code skeleton
    _step(2, "CEO delegates code writing to internal Builder agent")
    client = get_chat_client()
    _agent("Builder", "Writing landing page HTML/CSS skeleton ...")
    workflow = create_sequential_workflow(chat_client=client)
    internal_result = await workflow.run(
        "Write the HTML and CSS code for a responsive landing page for AgentOS"
    )
    internal_output = _extract_output_text(internal_result.get_outputs())
    _ok("Builder produced code skeleton")
    print()

    # 3 — Start external designer and run hiring workflow
    _step(3, "Starting external designer agent")
    async with _run_designer_server() as _server:
        _ok(f"Designer agent running on port {DESIGNER_PORT}")
        print()

        # 4 — CEO discovers external agents
        _step(4, "CEO discovers external agents in marketplace")
        from src.workflows.hiring import discover_external_agents
        candidates = discover_external_agents("design")
        for c in candidates:
            _agent(c.name, f"Skills: {', '.join(c.skills)} — Price: {c.price_per_call}")
        _ok(f"Found {len(candidates)} external agent(s)")
        print()

        # 5 — CEO hires designer and sends task
        _step(5, "CEO hires designer agent + sends task via A2A")
        hiring_result = await run_hiring_workflow(
            task_id=f"{TASK_ID}-design",
            task_description="Create a professional design specification for an AgentOS landing page",
            required_skills=["design", "ui", "landing-page"],
            budget_usd=BUDGET_USD,
            capability_query="design",
        )
        if hiring_result.status == "completed":
            _ok(f"Designer completed task")
            _agent("designer-ext-001", "Delivered design specifications")
        else:
            print(f"  {_C.RED}Hiring failed: {hiring_result.status}{_C.RESET}")
        print()

        # 6 — Payment recorded
        _step(6, "Payment recorded via x402 ledger")
        if hiring_result.payment:
            _money(f"Paid {hiring_result.payment['amount_usdc']:.2f} USDC to {hiring_result.payment['to_agent']}")
            _money(f"Transaction: {hiring_result.payment['tx_id']}")
        print()

    # 7 — Aggregate results
    _step(7, "CEO aggregates results from Builder + Designer")
    _agent("CEO", "Combining code from Builder with design from Designer")
    _ok("Task complete: landing page with professional design")
    print()

    # 8 — Budget summary
    _step(8, "Budget summary")
    # Show combined budgets
    b1 = ledger.get_budget(TASK_ID)
    b2 = ledger.get_budget(f"{TASK_ID}-design")
    total_allocated = (b1.allocated if b1 else 0) + (b2.allocated if b2 else 0)
    total_spent = (b1.spent if b1 else 0) + (b2.spent if b2 else 0)
    _info(f"Allocated : ${total_allocated:.2f} USDC")
    _info(f"Spent     : ${total_spent:.2f} USDC")
    _info(f"Remaining : ${total_allocated - total_spent:.2f} USDC")
    print()

    elapsed = time.monotonic() - t0

    # 9 — Save output
    _step(9, "Saving output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "agent_hiring_result.txt"

    output_parts = [
        "=" * 60,
        "AgentOS Demo: External Agent Hiring",
        "=" * 60,
        "",
        "--- Internal Builder Output ---",
        internal_output,
        "",
        "--- External Designer Deliverable ---",
    ]
    if hiring_result.external_result:
        deliverable = hiring_result.external_result.get("deliverable", "")
        output_parts.append(deliverable)
    output_parts.extend([
        "",
        "--- Payment ---",
        f"Amount: ${hiring_result.payment['amount_usdc']:.2f} USDC" if hiring_result.payment else "No payment",
        f"Agent: {hiring_result.payment['to_agent']}" if hiring_result.payment else "",
        f"TX: {hiring_result.payment['tx_id']}" if hiring_result.payment else "",
        "",
        f"Total elapsed: {elapsed:.2f}s",
    ])
    output_text = "\n".join(output_parts)
    out_path.write_text(output_text, encoding="utf-8")
    _ok(f"Saved to {out_path}")

    _header("Demo Complete")

    result = {
        "task": TASK_DESCRIPTION,
        "workflow": "hiring",
        "status": hiring_result.status,
        "internal_output": internal_output,
        "hiring_result": asdict(hiring_result),
        "budget": {
            "allocated": total_allocated,
            "spent": total_spent,
            "remaining": total_allocated - total_spent,
        },
        "elapsed_s": round(elapsed, 3),
        "output": output_text,
    }
    return result


if __name__ == "__main__":
    result = asyncio.run(run_agent_hiring_scenario())
    print(f"\nFinal status: {result['status']}")
