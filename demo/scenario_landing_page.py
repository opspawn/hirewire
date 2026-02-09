"""Demo Scenario: Build a landing page for AgentOS.

Submits a task to the CEO agent that requires both research and building,
running through the sequential workflow: Research -> CEO -> Builder.

Works with both mock and ollama providers (controlled by MODEL_PROVIDER env var).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import get_chat_client, get_settings
from src.mcp_servers.payment_hub import ledger
from src.workflows.sequential import create_sequential_workflow, _extract_output_text


TASK_DESCRIPTION = (
    "Research best practices for modern SaaS landing pages, then build a "
    "responsive HTML landing page for AgentOS - an AI agent operating system"
)

TASK_ID = "demo-landing-page"
BUDGET_USD = 5.0

OUTPUT_DIR = Path(__file__).parent / "output"


# -- ANSI helpers (no external deps) --

class _C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
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
    print(f"  {_C.GREEN}✓{_C.RESET} {text}")


def _info(text: str) -> None:
    print(f"  {_C.DIM}{text}{_C.RESET}")


async def run_landing_page_scenario() -> dict:
    """Run the landing page demo scenario.

    Returns:
        Dict with keys: task, workflow, status, output, budget, elapsed_s
    """
    _header("AgentOS Demo: Build a Landing Page")

    provider = get_settings().model_provider.value
    _info(f"Model provider: {provider}")
    _info(f"Task: {TASK_DESCRIPTION}")
    print()

    # 1 — Budget allocation
    _step(1, "Allocating budget")
    ledger.allocate_budget(TASK_ID, BUDGET_USD)
    budget = ledger.get_budget(TASK_ID)
    _ok(f"Budget: ${budget.allocated:.2f} USDC allocated")
    print()

    # 2 — Create agents and workflow
    _step(2, "Creating agents")
    client = get_chat_client()
    _agent("Research", "Web search, data analysis, reports")
    _agent("CEO", "Task analysis, budget management, delegation")
    _agent("Builder", "Code writing, testing, deployment")
    print()

    _step(3, "Building sequential workflow (Research -> CEO -> Builder)")
    workflow = create_sequential_workflow(chat_client=client)
    _ok("Workflow created")
    print()

    # 3 — Run workflow
    _step(4, "Executing workflow")
    t0 = time.monotonic()

    _agent("Research", "Gathering landing page best practices …")
    result = await workflow.run(TASK_DESCRIPTION)
    elapsed = time.monotonic() - t0

    outputs = result.get_outputs()
    output_text = _extract_output_text(outputs)

    _agent("CEO", "Analyzing research and creating execution plan …")
    _agent("Builder", "Implementing landing page …")
    _ok(f"Workflow complete in {elapsed:.2f}s")
    print()

    # 4 — Budget summary
    _step(5, "Budget summary")
    budget = ledger.get_budget(TASK_ID)
    _info(f"Allocated : ${budget.allocated:.2f} USDC")
    _info(f"Spent     : ${budget.spent:.2f} USDC")
    _info(f"Remaining : ${budget.remaining:.2f} USDC")
    print()

    # 5 — Save output
    _step(6, "Saving output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "landing_page_result.txt"
    out_path.write_text(output_text, encoding="utf-8")
    _ok(f"Saved to {out_path}")

    _header("Demo Complete")

    return {
        "task": TASK_DESCRIPTION,
        "workflow": "sequential",
        "status": str(result.get_final_state()),
        "output": output_text,
        "budget": {
            "allocated": budget.allocated,
            "spent": budget.spent,
            "remaining": budget.remaining,
        },
        "elapsed_s": round(elapsed, 3),
    }


if __name__ == "__main__":
    result = asyncio.run(run_landing_page_scenario())
    print(f"\nFinal status: {result['status']}")
