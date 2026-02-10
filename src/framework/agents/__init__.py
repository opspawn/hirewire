"""Pre-built agents for Microsoft Agent Framework integration.

Provides ready-made agents that demonstrate multi-agent orchestration:
- Researcher: Web search and information synthesis
- Analyst: Data analysis and reasoning
- Executor: Task execution and API calls
"""

from src.framework.agents.researcher import create_researcher_agent
from src.framework.agents.analyst import create_analyst_agent
from src.framework.agents.executor import create_executor_agent

__all__ = [
    "create_researcher_agent",
    "create_analyst_agent",
    "create_executor_agent",
]
