"""Builder Agent - Writes code, runs tests, and deploys services.

Uses ChatAgent with tools for GitHub operations and deployment.
Supports handoff pattern for receiving work from CEO agent.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from agent_framework import ChatAgent, tool

from src.config import get_chat_client

BUILDER_INSTRUCTIONS = """You are the Builder Agent of AgentOS.

Your responsibilities:
1. **Write Code**: Implement features, fix bugs, and refactor code based on task specs.
2. **Run Tests**: Execute test suites and verify code quality.
3. **Deploy**: Push code to repositories and deploy to hosting platforms.
4. **Report**: Provide detailed summaries of what was built and any issues found.

When you receive a task:
1. Analyze the requirements
2. Write or modify the necessary code
3. Run tests to verify correctness
4. Deploy if requested
5. Report back with: files changed, tests passed/failed, deployment status

You have access to:
- GitHub operations (commit, push, create PR)
- Deployment tools (restart services, deploy to cloud)
- Code analysis tools

Always write clean, typed, well-documented code. Follow existing project patterns.
"""


@tool(name="github_commit", description="Commit changes to a GitHub repository")
async def github_commit(
    repo: str,
    branch: str,
    message: str,
    files: list[str] | None = None,
) -> dict[str, Any]:
    """Commit changes to a local git repository using real git commands.

    Args:
        repo: Local path to the git repository.
        branch: Branch name to commit on.
        message: Commit message.
        files: Optional list of specific files to add. If None, adds all changes.
    """
    try:
        repo_path = os.path.expanduser(repo)
        if not os.path.isdir(repo_path):
            return {"status": "error", "error": f"Repository path not found: {repo_path}"}

        # Check out the target branch
        proc = await asyncio.create_subprocess_exec(
            "git", "checkout", branch,
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            # Try creating the branch if it doesn't exist
            proc = await asyncio.create_subprocess_exec(
                "git", "checkout", "-b", branch,
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                return {"status": "error", "error": f"Failed to checkout branch: {stderr.decode().strip()}"}

        # Stage files
        if files:
            add_cmd = ["git", "add"] + files
        else:
            add_cmd = ["git", "add", "-A"]

        proc = await asyncio.create_subprocess_exec(
            *add_cmd,
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            return {"status": "error", "error": f"git add failed: {stderr.decode().strip()}"}

        # Commit
        proc = await asyncio.create_subprocess_exec(
            "git", "commit", "-m", message, "--allow-empty",
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            output = stderr.decode().strip()
            if "nothing to commit" in output or "nothing to commit" in stdout.decode():
                return {"status": "no_changes", "repo": repo, "branch": branch, "message": message}
            return {"status": "error", "error": f"git commit failed: {output}"}

        # Get commit SHA
        proc = await asyncio.create_subprocess_exec(
            "git", "log", "-1", "--format=%H",
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        commit_sha = stdout.decode().strip()

        # Get list of files committed (handle first commit with no parent)
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--name-only", "HEAD~1", "HEAD",
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            files_committed = [f for f in stdout.decode().strip().split("\n") if f]
        else:
            # First commit (no parent) — use ls-tree to list files
            proc = await asyncio.create_subprocess_exec(
                "git", "ls-tree", "--name-only", "-r", "HEAD",
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            files_committed = [f for f in stdout.decode().strip().split("\n") if f]

        return {
            "status": "committed",
            "repo": repo,
            "branch": branch,
            "message": message,
            "files_committed": files_committed,
            "commit_sha": commit_sha,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool(name="deploy_service", description="Deploy a service to the target environment")
async def deploy_service(
    service_name: str,
    target: str = "local",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deploy a service.

    Placeholder - will integrate with real deployment systems.
    """
    return {
        "status": "deployed",
        "service": service_name,
        "target": target,
        "url": f"http://localhost:8080/{service_name}" if target == "local" else f"https://{service_name}.azurewebsites.net",
        "config": config or {},
    }


@tool(name="run_tests", description="Run test suite for a project")
async def run_tests(
    project_path: str,
    test_pattern: str = "tests/",
) -> dict[str, Any]:
    """Run project tests using pytest.

    Args:
        project_path: Path to the project root.
        test_pattern: Path or pattern for test discovery (default: tests/).
    """
    try:
        proj = os.path.expanduser(project_path)
        if not os.path.isdir(proj):
            return {"status": "error", "project": project_path, "error": f"Path not found: {proj}"}

        # Look for a venv pytest, fall back to system pytest
        venv_pytest = os.path.join(proj, ".venv", "bin", "pytest")
        pytest_cmd = venv_pytest if os.path.isfile(venv_pytest) else "pytest"

        # test_pattern can be absolute or relative to project_path
        if os.path.isabs(test_pattern):
            test_target = test_pattern
        else:
            test_target = os.path.join(proj, test_pattern)

        proc = await asyncio.create_subprocess_exec(
            pytest_cmd, test_target, "-v", "--tb=short",
            cwd=proj,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        output = stdout_bytes.decode()

        # Parse pytest output
        tests_run = 0
        tests_passed = 0
        tests_failed = 0
        duration_ms = 0

        # Look for the final summary line, e.g. "= 5 passed, 2 failed in 1.23s ="
        # Use a specific pattern that matches the content (passed/failed/error + in Xs)
        summary_match = re.search(
            r"=+\s*((?:\d+\s+\w+(?:,\s*)?)+\s+in\s+[\d.]+s)\s*=+",
            output,
        )
        if summary_match:
            summary_line = summary_match.group(1)
            passed_m = re.search(r"(\d+)\s+passed", summary_line)
            failed_m = re.search(r"(\d+)\s+failed", summary_line)
            error_m = re.search(r"(\d+)\s+error", summary_line)
            duration_m = re.search(r"in\s+([\d.]+)s", summary_line)

            if passed_m:
                tests_passed = int(passed_m.group(1))
            if failed_m:
                tests_failed = int(failed_m.group(1))
            if error_m:
                tests_failed += int(error_m.group(1))
            if duration_m:
                duration_ms = int(float(duration_m.group(1)) * 1000)
            tests_run = tests_passed + tests_failed

        status = "passed" if proc.returncode == 0 else "failed"

        return {
            "status": status,
            "project": project_path,
            "tests_run": tests_run,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "duration_ms": duration_ms,
            "output": output[-2000:] if len(output) > 2000 else output,
        }
    except Exception as e:
        return {"status": "error", "project": project_path, "error": str(e)}


def create_builder_agent(chat_client=None) -> ChatAgent:
    """Create and return the Builder agent.

    Args:
        chat_client: Optional ChatClientProtocol instance. If None, creates one
                     from environment config.
    """
    if chat_client is None:
        chat_client = get_chat_client()

    return ChatAgent(
        chat_client=chat_client,
        name="Builder",
        description="Code builder agent that writes, tests, and deploys software",
        instructions=BUILDER_INSTRUCTIONS,
        tools=[github_commit, deploy_service, run_tests],
    )
