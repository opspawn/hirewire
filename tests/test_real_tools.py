"""Tests for real (non-placeholder) tool implementations.

Tests the actual logic in agent tools: task analysis parsing,
real ledger budget checking, git subprocess calls, pytest runner,
and DuckDuckGo web search.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.ceo_agent import analyze_task, check_budget
from src.agents.builder_agent import github_commit, run_tests
from src.agents.research_agent import web_search
from src.mcp_servers.payment_hub import PaymentLedger, ledger


# ──────────────────────────────────────────────
# CEO: analyze_task
# ──────────────────────────────────────────────

class TestAnalyzeTask:
    """Test intelligent task analysis with keyword matching."""

    @pytest.mark.asyncio
    async def test_research_only_task(self):
        result = await analyze_task("Search for the best Python frameworks and compare them")
        assert result["task_type"] == "research"
        assert len(result["subtasks"]) == 1
        assert result["subtasks"][0]["agent"] == "research"

    @pytest.mark.asyncio
    async def test_build_only_task(self):
        result = await analyze_task("Build a REST API with FastAPI")
        assert result["task_type"] == "build"
        assert len(result["subtasks"]) == 1
        assert result["subtasks"][0]["agent"] == "builder"

    @pytest.mark.asyncio
    async def test_research_and_build_task(self):
        result = await analyze_task("Research the best database options and build a data pipeline")
        assert result["task_type"] == "research+build"
        assert len(result["subtasks"]) == 2
        assert result["execution_order"] == "sequential"
        agents = [s["agent"] for s in result["subtasks"]]
        assert "research" in agents
        assert "builder" in agents

    @pytest.mark.asyncio
    async def test_general_task_defaults_to_both(self):
        result = await analyze_task("handle this thing for me please")
        assert result["task_type"] == "general"
        assert len(result["subtasks"]) == 2

    @pytest.mark.asyncio
    async def test_complexity_simple(self):
        result = await analyze_task("build a hello world app")
        assert result["complexity"] == "simple"
        assert result["estimated_cost"] > 0

    @pytest.mark.asyncio
    async def test_complexity_scales_with_words(self):
        short = await analyze_task("build an app")
        long_desc = "research and build " + " ".join(["word"] * 50)
        long_result = await analyze_task(long_desc)
        assert long_result["estimated_cost"] > short["estimated_cost"]

    @pytest.mark.asyncio
    async def test_status_is_planned(self):
        result = await analyze_task("create something")
        assert result["status"] == "planned"

    @pytest.mark.asyncio
    async def test_original_task_preserved(self):
        desc = "Deploy the microservice to production"
        result = await analyze_task(desc)
        assert result["original_task"] == desc


# ──────────────────────────────────────────────
# CEO: check_budget
# ──────────────────────────────────────────────

class TestCheckBudget:
    """Test real ledger integration for budget checking."""

    @pytest.fixture(autouse=True)
    def _reset_ledger(self):
        """Clear ledger state between tests."""
        ledger._budgets.clear()
        ledger._transactions.clear()
        ledger._tx_counter = 0
        yield
        ledger._budgets.clear()
        ledger._transactions.clear()
        ledger._tx_counter = 0

    @pytest.mark.asyncio
    async def test_no_budget_returns_message(self):
        result = await check_budget("nonexistent_task")
        assert result["allocated"] == 0.0
        assert "message" in result
        assert "No budget" in result["message"]

    @pytest.mark.asyncio
    async def test_allocated_budget_returned(self):
        ledger.allocate_budget("task_123", 5.0)
        result = await check_budget("task_123")
        assert result["allocated"] == 5.0
        assert result["spent"] == 0.0
        assert result["remaining"] == 5.0
        assert result["currency"] == "USDC"

    @pytest.mark.asyncio
    async def test_spent_budget_reflected(self):
        ledger.allocate_budget("task_456", 10.0)
        ledger.record_payment("ceo", "builder", 3.5, "task_456")
        result = await check_budget("task_456")
        assert result["allocated"] == 10.0
        assert result["spent"] == 3.5
        assert result["remaining"] == 6.5


# ──────────────────────────────────────────────
# Builder: github_commit
# ──────────────────────────────────────────────

class TestGithubCommit:
    """Test real git subprocess integration."""

    @pytest.mark.asyncio
    async def test_nonexistent_repo_returns_error(self):
        result = await github_commit(
            repo="/tmp/nonexistent_repo_xyz",
            branch="main",
            message="test commit",
        )
        assert result["status"] == "error"
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_real_git_commit_in_temp_repo(self):
        """Create a temp git repo, add a file, and commit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Init a git repo with initial branch "main"
            proc = await asyncio.create_subprocess_exec(
                "git", "init", "-b", "main", tmpdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            # Configure git user for the temp repo
            for cmd in [
                ["git", "config", "user.email", "test@test.com"],
                ["git", "config", "user.name", "Test"],
            ]:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, cwd=tmpdir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()

            # Create a file
            test_file = os.path.join(tmpdir, "hello.txt")
            with open(test_file, "w") as f:
                f.write("hello world\n")

            # Use our tool to commit
            result = await github_commit(
                repo=tmpdir,
                branch="main",
                message="Initial commit",
                files=["hello.txt"],
            )

            assert result["status"] == "committed"
            assert len(result["commit_sha"]) == 40  # full SHA
            assert "hello.txt" in result["files_committed"]


# ──────────────────────────────────────────────
# Builder: run_tests
# ──────────────────────────────────────────────

class TestRunTests:
    """Test real pytest runner."""

    @pytest.mark.asyncio
    async def test_nonexistent_path_returns_error(self):
        result = await run_tests(project_path="/tmp/nonexistent_project_xyz")
        assert result["status"] == "error"
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_run_own_tests(self):
        """Run the project's own test suite (agents only, fast)."""
        result = await run_tests(
            project_path="/home/agent/projects/ms-agent-framework-hackathon",
            test_pattern="tests/test_agents.py",
        )
        assert result["status"] == "passed"
        assert result["tests_run"] > 0
        assert result["tests_passed"] > 0
        assert result["tests_failed"] == 0
        assert result["duration_ms"] > 0


# ──────────────────────────────────────────────
# Research: web_search
# ──────────────────────────────────────────────

class TestWebSearch:
    """Test DuckDuckGo search integration."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """Test that a real search returns actual results."""
        result = await web_search("Python programming language", max_results=3)
        assert result["query"] == "Python programming language"
        assert result["total_results"] > 0
        assert len(result["results"]) > 0

        # Each result should have title, url, snippet
        first = result["results"][0]
        assert "title" in first
        assert "url" in first
        assert "snippet" in first

    @pytest.mark.asyncio
    async def test_search_respects_max_results(self):
        result = await web_search("FastAPI framework", max_results=2)
        assert result["total_results"] <= 2

    @pytest.mark.asyncio
    async def test_search_error_handling(self):
        """Test that import errors are handled gracefully."""
        with patch.dict("sys.modules", {"duckduckgo_search": None}):
            # Force ImportError by patching the import
            with patch("src.agents.research_agent.web_search", wraps=web_search):
                # We can't easily force ImportError on already-imported module,
                # so we test the exception path with a mock
                pass

        # Instead, test that an exception in DDGS is caught
        with patch("ddgs.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.text.side_effect = RuntimeError("Network error")
            mock_ddgs.return_value = mock_instance

            result = await web_search("test query")
            assert result["total_results"] == 0
            assert "error" in result
            assert "Search failed" in result["error"]
