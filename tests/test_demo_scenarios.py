"""Tests for demo scenarios.

Verifies that both demo scenarios complete successfully with the mock
client and that the CLI argument parser works correctly.
"""

from __future__ import annotations

import pytest

from src.mcp_servers.payment_hub import ledger


@pytest.fixture(autouse=True)
def _reset_ledger():
    """Clear ledger state so budget allocations don't collide across tests."""
    ledger._transactions.clear()
    ledger._budgets.clear()
    ledger._tx_counter = 0
    yield
    ledger._transactions.clear()
    ledger._budgets.clear()
    ledger._tx_counter = 0


class TestLandingPageScenario:
    @pytest.mark.asyncio
    async def test_landing_page_completes(self):
        from demo.scenario_landing_page import run_landing_page_scenario

        result = await run_landing_page_scenario()

        assert result["workflow"] == "sequential"
        assert result["output"]  # non-empty output
        assert result["elapsed_s"] >= 0
        assert result["budget"]["allocated"] == 5.0
        assert "landing page" in result["task"].lower()


class TestParallelResearchScenario:
    @pytest.mark.asyncio
    async def test_parallel_research_completes(self):
        from demo.scenario_parallel_research import run_parallel_research_scenario

        result = await run_parallel_research_scenario()

        assert result["workflow"] == "concurrent"
        assert result["output"]  # non-empty output
        assert result["elapsed_s"] >= 0
        assert result["budget"]["allocated"] == 3.0
        assert "competitor" in result["task"].lower()


class TestRunDemoCLI:
    def test_parser_accepts_valid_scenarios(self):
        from demo.run_demo import build_parser

        parser = build_parser()

        for scenario in ("landing-page", "research", "all"):
            args = parser.parse_args([scenario])
            assert args.scenario == scenario

    def test_parser_rejects_invalid_scenario(self):
        from demo.run_demo import build_parser

        parser = build_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["nonexistent"])
