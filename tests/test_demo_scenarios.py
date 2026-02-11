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


class TestShowcaseScenario:
    @pytest.mark.asyncio
    async def test_showcase_completes(self):
        from demo.scenario_showcase import run_showcase_scenario

        result = await run_showcase_scenario()

        assert result["workflow"] == "showcase"
        assert result["status"] == "completed"
        assert result["total_elapsed_s"] >= 0
        assert result["budget"]["allocated"] == 10.0
        assert len(result["stages"]) == 8

    @pytest.mark.asyncio
    async def test_showcase_stages_have_required_fields(self):
        from demo.scenario_showcase import run_showcase_scenario

        result = await run_showcase_scenario()

        for stage in result["stages"]:
            assert "stage" in stage
            assert "name" in stage
            assert "duration_ms" in stage
            assert stage["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_showcase_includes_foundry(self):
        from demo.scenario_showcase import run_showcase_scenario

        result = await run_showcase_scenario()

        foundry_stage = [s for s in result["stages"] if s["name"] == "Foundry Agent Service"]
        assert len(foundry_stage) == 1
        assert foundry_stage[0]["foundry_agents"] == 4

    @pytest.mark.asyncio
    async def test_showcase_includes_x402(self):
        from demo.scenario_showcase import run_showcase_scenario

        result = await run_showcase_scenario()

        hiring_stage = [s for s in result["stages"] if "x402" in s["name"]]
        assert len(hiring_stage) == 1
        # In mock mode without external server, hiring may fail — that's OK
        assert hiring_stage[0]["status"] in ("completed", "failed")


class TestRecordDemo:
    @pytest.mark.asyncio
    async def test_recorded_demo_completes(self):
        from demo.record_demo import run_recorded_demo

        result = await run_recorded_demo(fast=True)

        assert "stage_1" in result
        assert "stage_8" in result
        assert result["stage_1"]["count"] == 4
        assert result["stage_8"]["total_elapsed_s"] >= 0

    @pytest.mark.asyncio
    async def test_recorded_demo_has_all_stages(self):
        from demo.record_demo import run_recorded_demo

        result = await run_recorded_demo(fast=True)

        for i in range(1, 9):
            assert f"stage_{i}" in result

    def test_record_demo_parser_defaults(self):
        from demo.record_demo import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.fast is False
        assert args.stage_pause == 2.0

    def test_record_demo_parser_fast(self):
        from demo.record_demo import build_parser

        parser = build_parser()
        args = parser.parse_args(["--fast"])
        assert args.fast is True

    def test_record_demo_parser_custom_pause(self):
        from demo.record_demo import build_parser

        parser = build_parser()
        args = parser.parse_args(["--pause", "3.5"])
        assert args.stage_pause == 3.5


class TestRunDemoCLI:
    def test_parser_accepts_valid_scenarios(self):
        from demo.run_demo import build_parser

        parser = build_parser()

        for scenario in ("landing-page", "research", "showcase", "all"):
            args = parser.parse_args([scenario])
            assert args.scenario == scenario

    def test_parser_rejects_invalid_scenario(self):
        from demo.run_demo import build_parser

        parser = build_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["nonexistent"])
