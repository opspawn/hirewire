"""Shared test configuration.

Forces mock model provider for all tests so integration tests
don't depend on Ollama or Azure being available.
"""

import os

import pytest

# Force mock provider before any module imports config
os.environ["MODEL_PROVIDER"] = "mock"


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Clear cached settings so each test gets fresh config."""
    from src.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
