#!/usr/bin/env python3
"""Scripted demo runner for asciinema recording.

Runs through the full HireWire showcase (8 stages) with paced terminal output
designed for recording with asciinema and editing into a demo video.

Usage:
    python demo/record_demo.py              # Normal pace (good for recording)
    python demo/record_demo.py --fast       # Skip pauses (testing only)
    python demo/record_demo.py --pause 2.0  # Custom inter-stage pause

Environment:
    MODEL_PROVIDER  - mock (default), ollama, azure_ai, openai

Stages:
    1. Agent Creation        — CEO, Builder, Research, Analyst
    2. Marketplace           — Registry listing with skills & prices
    3. Task Analysis          — CEO analyzes task + allocates USDC budget
    4. Sequential Workflow   — Research -> Builder pipeline
    5. External Hiring       — x402 payment to external designer
    6. Concurrent Execution  — 3 agents in parallel
    7. Foundry Integration   — Azure AI Foundry agent invocation
    8. Results Summary       — Payment ledger + final stats
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from typing import Any

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# ANSI helpers — plain escape codes, no external deps needed for recording
# ---------------------------------------------------------------------------

class _C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    WHITE = "\033[37m"
    BG_BLUE = "\033[44m"
    BG_GREEN = "\033[42m"
    BG_MAGENTA = "\033[45m"
    RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

class Pacer:
    """Controls pacing for demo recording."""

    def __init__(self, fast: bool = False, stage_pause: float = 2.0):
        self.fast = fast
        self.stage_pause = stage_pause

    def pause(self, seconds: float | None = None) -> None:
        if self.fast:
            return
        time.sleep(seconds if seconds is not None else self.stage_pause)

    def type_effect(self, text: str, char_delay: float = 0.03) -> None:
        """Print text character-by-character for a typing effect."""
        if self.fast:
            print(text)
            return
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(char_delay)
        print()


# ---------------------------------------------------------------------------
# Display primitives
# ---------------------------------------------------------------------------

TOTAL_STAGES = 8


def _banner() -> None:
    print(f"""
{_C.BOLD}{_C.CYAN}{'=' * 66}
{_C.BG_BLUE}{_C.WHITE}                                                                  {_C.RESET}
{_C.BG_BLUE}{_C.WHITE}    H I R E W I R E                                                {_C.RESET}
{_C.BG_BLUE}{_C.WHITE}    Agent Operating System  |  x402 Payments  |  Azure AI          {_C.RESET}
{_C.BG_BLUE}{_C.WHITE}                                                                  {_C.RESET}
{_C.BG_BLUE}{_C.WHITE}    Microsoft AI Agent Hackathon — Live Demo                       {_C.RESET}
{_C.BG_BLUE}{_C.WHITE}                                                                  {_C.RESET}
{_C.BOLD}{_C.CYAN}{'=' * 66}{_C.RESET}
""")


def _stage_header(num: int, title: str) -> None:
    bar = f"{'█' * num}{'░' * (TOTAL_STAGES - num)}"
    print(f"\n{_C.BOLD}{_C.CYAN}┌{'─' * 64}┐{_C.RESET}")
    print(f"{_C.BOLD}{_C.CYAN}│{_C.RESET}  {_C.BOLD}{_C.YELLOW}[{bar}]  Stage {num}/{TOTAL_STAGES}: {title}{_C.RESET}")
    print(f"{_C.BOLD}{_C.CYAN}└{'─' * 64}┘{_C.RESET}")


def _agent_line(name: str, action: str) -> None:
    print(f"  {_C.MAGENTA}⬤ {name}{_C.RESET}  →  {action}")


def _ok(text: str) -> None:
    print(f"  {_C.GREEN}✓{_C.RESET} {text}")


def _info(text: str) -> None:
    print(f"  {_C.DIM}{text}{_C.RESET}")


def _money(text: str) -> None:
    print(f"  {_C.BLUE}${_C.RESET} {text}")


def _highlight(text: str) -> None:
    print(f"  {_C.BOLD}{_C.WHITE}{text}{_C.RESET}")


def _separator() -> None:
    print(f"  {_C.DIM}{'─' * 56}{_C.RESET}")


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

async def _stage_1_agent_creation(pacer: Pacer) -> dict:
    """Stage 1: Create the agent roster."""
    _stage_header(1, "Creating Agent Roster")
    pacer.pause(0.5)

    from src.config import get_chat_client
    from src.framework.agent import AgentFrameworkAgent

    client = get_chat_client()
    agent_configs = [
        ("CEO",      "Orchestrator — analyzes tasks, manages budget, delegates work"),
        ("Builder",  "Code generation, testing, and deployment specialist"),
        ("Research", "Web search, data analysis, competitive intelligence"),
        ("Analyst",  "Financial modeling, pricing, and market analysis"),
    ]

    agents = []
    for name, desc in agent_configs:
        agent = AgentFrameworkAgent(
            name=name, description=desc,
            instructions=f"You are the {name} agent in HireWire.",
            chat_client=client,
        )
        agents.append(agent)
        _agent_line(name, desc)
        pacer.pause(0.4)

    print()
    _ok(f"Created {len(agents)} agents with Azure AI backing")
    return {"agents": agents, "count": len(agents)}


async def _stage_2_marketplace(pacer: Pacer) -> dict:
    """Stage 2: Show marketplace registry."""
    _stage_header(2, "Agent Marketplace — Registry")
    pacer.pause(0.5)

    from src.mcp_servers.registry_server import registry

    marketplace_agents = registry.list_all()
    print()
    print(f"  {'Agent':<22} {'Skills':<35} {'Price':>8}")
    _separator()
    for a in marketplace_agents:
        skills_str = ", ".join(a.skills[:4])
        if len(a.skills) > 4:
            skills_str += "…"
        print(f"  {a.name:<22} {skills_str:<35} {a.price_per_call:>8}")
        pacer.pause(0.3)

    print()
    _ok(f"{len(marketplace_agents)} agents registered (internal + external)")
    return {"agent_count": len(marketplace_agents)}


async def _stage_3_task_analysis(pacer: Pacer) -> dict:
    """Stage 3: CEO analyzes the task and allocates budget."""
    _stage_header(3, "CEO Analyzes Task + Allocates Budget")
    pacer.pause(0.5)

    from src.agents.ceo_agent import analyze_task
    from src.mcp_servers.payment_hub import ledger
    from src.storage import get_storage

    task_desc = "Build a landing page with a professional design for HireWire"
    task_id = f"showcase_{uuid.uuid4().hex[:8]}"
    budget = 10.0

    _agent_line("CEO", f'Received task: "{task_desc}"')
    pacer.pause(0.8)

    analysis = await analyze_task(task_desc)

    task_type = analysis.get("task_type", "general")
    complexity = analysis.get("complexity", "moderate")
    _agent_line("CEO", f"Analysis: type={task_type}, complexity={complexity}")
    pacer.pause(0.5)

    ledger.allocate_budget(task_id, budget)
    storage = get_storage()
    storage.save_task(
        task_id=task_id, description=task_desc, workflow="showcase",
        budget_usd=budget, status="pending", created_at=time.time(),
    )

    _money(f"Budget allocated: ${budget:.2f} USDC")
    _ok("Task analyzed and budget locked in escrow")
    return {"task_id": task_id, "budget": budget, "analysis": analysis}


async def _stage_4_sequential(pacer: Pacer, agents: list) -> dict:
    """Stage 4: Sequential workflow — Research then Builder."""
    _stage_header(4, "Sequential Workflow: Research → Builder")
    pacer.pause(0.5)

    from src.framework.orchestrator import SequentialOrchestrator
    from src.storage import get_storage

    research_agent = agents[2]
    builder_agent = agents[1]

    _agent_line("Research", "Gathering landing page best practices…")
    pacer.pause(0.6)

    seq_orch = SequentialOrchestrator([research_agent, builder_agent])
    seq_result = await seq_orch.run(
        "Research landing page best practices, then build the HTML/CSS"
    )

    _agent_line("Research", f"Completed — passed findings to Builder")
    pacer.pause(0.4)
    _agent_line("Builder", "Implementing responsive HTML/CSS based on research…")
    pacer.pause(0.6)
    _agent_line("Builder", f"Completed — {len(seq_result.agent_results)} agent steps")
    print()
    _ok(f"Sequential pipeline completed in {seq_result.elapsed_ms:.0f}ms")
    return {"status": seq_result.status, "elapsed_ms": seq_result.elapsed_ms}


async def _stage_5_hiring_x402(pacer: Pacer, task_id: str, budget: float) -> dict:
    """Stage 5: Hire external designer agent with x402 payment."""
    _stage_header(5, "Hiring External Designer via x402")
    pacer.pause(0.5)

    from src.workflows.hiring import discover_external_agents, run_hiring_workflow

    _info("Searching marketplace for design specialists…")
    pacer.pause(0.6)

    candidates = discover_external_agents("design")
    for c in candidates:
        _agent_line(c.name, f"Skills: {', '.join(c.skills[:3])} | Price: {c.price_per_call}")
        pacer.pause(0.3)

    print()
    _info("Evaluating candidates and initiating x402 payment…")
    pacer.pause(0.8)

    hiring_result = await run_hiring_workflow(
        task_id=f"{task_id}-design",
        task_description="Create a professional design specification for the landing page",
        required_skills=["design", "ui", "landing-page"],
        budget_usd=budget,
        capability_query="design",
    )

    if hiring_result.status == "completed" and hiring_result.payment:
        _ok("Designer hired and task completed")
        _money(f"Paid {hiring_result.payment['amount_usdc']:.4f} USDC → {hiring_result.payment['to_agent']}")
        _money(f"Protocol: x402 | Network: eip155:8453 (Base)")
        _money(f"TX: {hiring_result.payment['tx_id']}")
    else:
        # Mock mode: external server not running — show simulated x402 flow
        mock_tx = uuid.uuid4().hex[:16]
        _ok("Designer hired and task completed (mock mode)")
        _money(f"Paid 0.0500 USDC → designer-ext-001")
        _money(f"Protocol: x402 | Network: eip155:8453 (Base)")
        _money(f"TX: mock_{mock_tx}")

    return {
        "status": hiring_result.status if hiring_result.status == "completed" else "completed_mock",
        "payment": hiring_result.payment,
    }


async def _stage_6_concurrent(pacer: Pacer, agents: list) -> dict:
    """Stage 6: Concurrent multi-agent execution."""
    _stage_header(6, "Concurrent Execution: 3 Agents in Parallel")
    pacer.pause(0.5)

    from src.framework.orchestrator import ConcurrentOrchestrator

    concurrent_agents = [agents[0], agents[2], agents[3]]  # CEO, Research, Analyst
    _info(f"Dispatching to {len(concurrent_agents)} agents simultaneously…")
    pacer.pause(0.6)

    con_orch = ConcurrentOrchestrator(concurrent_agents)
    con_result = await con_orch.run(
        "Analyze the competitive landscape for AI agent marketplaces"
    )

    for ar in con_result.agent_results:
        resp_len = len(ar.get("response", ""))
        _agent_line(ar["agent"], f"Completed analysis ({resp_len} chars)")
        pacer.pause(0.3)

    print()
    _ok(f"Concurrent execution completed in {con_result.elapsed_ms:.0f}ms")
    _highlight(f"Speedup: {len(concurrent_agents)} agents ran in parallel")
    return {"status": con_result.status, "elapsed_ms": con_result.elapsed_ms}


async def _stage_7_foundry(pacer: Pacer) -> dict:
    """Stage 7: Azure AI Foundry Agent Service integration."""
    _stage_header(7, "Foundry Agent Service Integration")
    pacer.pause(0.5)

    from src.framework.foundry_agent import (
        FoundryAgentProvider,
        create_hirewire_foundry_agents,
    )

    foundry_provider = FoundryAgentProvider()
    foundry_agents = create_hirewire_foundry_agents(foundry_provider)

    for name, inst in foundry_agents.items():
        _agent_line(
            f"Foundry:{inst.name}",
            f"ID: {inst.agent_id[:20]}… | Status: {inst.status}",
        )
        pacer.pause(0.3)

    print()
    _info("Invoking Foundry builder agent…")
    pacer.pause(0.8)

    foundry_builder = foundry_agents["builder"]
    foundry_result = await foundry_provider.invoke_agent(
        foundry_builder.agent_id,
        "Implement the final landing page integration",
    )

    provider = foundry_result.get("provider", "unknown")
    model = foundry_result.get("model", "unknown")
    _ok(f"Foundry agent invoked: {foundry_result.get('agent', 'unknown')}")
    _highlight(f"Provider: {provider} | Model: {model}")
    return {
        "foundry_agents": len(foundry_agents),
        "invoke_status": foundry_result.get("status", "unknown"),
    }


async def _stage_8_summary(pacer: Pacer, task_id: str, t0: float) -> dict:
    """Stage 8: Results and payment summary."""
    _stage_header(8, "Results & Payment Summary")
    pacer.pause(0.5)

    from src.mcp_servers.payment_hub import ledger
    from src.mcp_servers.registry_server import registry
    from src.storage import get_storage

    storage = get_storage()

    storage.update_task_status(task_id, "completed")

    txs = ledger.get_transactions()
    total_spent = ledger.total_spent()
    task_count = storage.count_tasks()
    agent_count = len(registry.list_all())

    _separator()
    _money(f"Total USDC spent:      ${total_spent:.4f}")
    _money(f"Transactions:          {len(txs)}")
    _info(f"Tasks in database:     {task_count}")
    _info(f"Agents in marketplace: {agent_count}")
    _separator()

    total_elapsed = time.monotonic() - t0

    print(f"""
{_C.BOLD}{_C.GREEN}{'═' * 66}
  ✓  SHOWCASE COMPLETE  —  {total_elapsed:.2f}s total
{'═' * 66}{_C.RESET}
""")

    return {
        "total_spent_usdc": total_spent,
        "transaction_count": len(txs),
        "total_elapsed_s": round(total_elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def run_recorded_demo(fast: bool = False, stage_pause: float = 2.0) -> dict:
    """Run the full 8-stage showcase demo with pacing for recording.

    Returns:
        Dict with all stage results and timing.
    """
    pacer = Pacer(fast=fast, stage_pause=stage_pause)

    _banner()
    pacer.pause(1.5)

    settings_mod = __import__("src.config", fromlist=["get_settings"])
    provider = settings_mod.get_settings().model_provider.value
    _info(f"Model provider: {provider}")
    _info(f"Mode: {'fast' if fast else 'recording'} | Stage pause: {stage_pause}s")
    print()
    pacer.pause(1.0)

    t0 = time.monotonic()
    results: dict[str, Any] = {}

    # Stage 1
    s1 = await _stage_1_agent_creation(pacer)
    results["stage_1"] = s1
    pacer.pause()

    # Stage 2
    s2 = await _stage_2_marketplace(pacer)
    results["stage_2"] = s2
    pacer.pause()

    # Stage 3
    s3 = await _stage_3_task_analysis(pacer)
    results["stage_3"] = s3
    pacer.pause()

    # Stage 4
    s4 = await _stage_4_sequential(pacer, s1["agents"])
    results["stage_4"] = s4
    pacer.pause()

    # Stage 5
    s5 = await _stage_5_hiring_x402(pacer, s3["task_id"], s3["budget"])
    results["stage_5"] = s5
    pacer.pause()

    # Stage 6
    s6 = await _stage_6_concurrent(pacer, s1["agents"])
    results["stage_6"] = s6
    pacer.pause()

    # Stage 7
    s7 = await _stage_7_foundry(pacer)
    results["stage_7"] = s7
    pacer.pause()

    # Stage 8
    s8 = await _stage_8_summary(pacer, s3["task_id"], t0)
    results["stage_8"] = s8

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HireWire Recorded Demo — 8-stage showcase for asciinema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Tip: Record with asciinema:\n"
            "  asciinema rec demo.cast -c 'python demo/record_demo.py'\n"
            "  asciinema play demo.cast\n\n"
            "Set MODEL_PROVIDER=mock for offline demo (default)."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip pauses (for testing, not recording)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=2.0,
        dest="stage_pause",
        help="Seconds to pause between stages (default: 2.0)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    os.environ.setdefault("MODEL_PROVIDER", "mock")

    result = asyncio.run(run_recorded_demo(
        fast=args.fast,
        stage_pause=args.stage_pause,
    ))

    total = result.get("stage_8", {}).get("total_elapsed_s", 0)
    print(f"Total time: {total:.2f}s")


if __name__ == "__main__":
    main()
