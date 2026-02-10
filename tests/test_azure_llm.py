"""Tests for the Azure OpenAI LLM provider.

All Azure API calls are mocked so tests run without credentials.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# Ensure mock provider is used (conftest sets this, but be explicit)
os.environ.setdefault("MODEL_PROVIDER", "mock")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _azure_env(monkeypatch):
    """Set Azure OpenAI environment variables for tests."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "test-key-12345")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    # Reset singleton
    import src.framework.azure_llm as mod
    mod._provider = None
    yield
    mod._provider = None


def _make_mock_response(content: str = "Hello!", model: str = "gpt-4o"):
    """Build a mock OpenAI ChatCompletion response object."""
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15

    message = MagicMock()
    message.content = content
    message.role = "assistant"

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# Tests — AzureLLMProvider
# ---------------------------------------------------------------------------


class TestAzureLLMProvider:
    """Unit tests for AzureLLMProvider."""

    def test_init_from_env(self, _azure_env):
        from src.framework.azure_llm import AzureLLMProvider

        provider = AzureLLMProvider()
        assert provider.endpoint == "https://test.openai.azure.com/"
        assert provider.api_key == "test-key-12345"
        assert provider.deployment == "gpt-4o"

    def test_init_explicit_args(self):
        from src.framework.azure_llm import AzureLLMProvider

        provider = AzureLLMProvider(
            endpoint="https://custom.openai.azure.com/",
            api_key="custom-key",
            deployment="gpt-35-turbo",
        )
        assert provider.endpoint == "https://custom.openai.azure.com/"
        assert provider.api_key == "custom-key"
        assert provider.deployment == "gpt-35-turbo"

    def test_chat_completion(self, _azure_env):
        from src.framework.azure_llm import AzureLLMProvider

        provider = AzureLLMProvider()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response("Test reply")
        provider._client = mock_client

        result = provider.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result["content"] == "Test reply"
        assert result["role"] == "assistant"
        assert result["model"] == "gpt-4o"
        assert result["usage"]["total_tokens"] == 15
        assert result["finish_reason"] == "stop"
        mock_client.chat.completions.create.assert_called_once()

    def test_generate(self, _azure_env):
        from src.framework.azure_llm import AzureLLMProvider

        provider = AzureLLMProvider()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response("Generated text")
        provider._client = mock_client

        text = provider.generate("Tell me a joke")
        assert text == "Generated text"

        # Verify system + user messages were sent
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Tell me a joke"

    def test_check_connection_success(self, _azure_env):
        from src.framework.azure_llm import AzureLLMProvider

        provider = AzureLLMProvider()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response("pong")
        provider._client = mock_client

        result = provider.check_connection()
        assert result["connected"] is True
        assert result["model"] == "gpt-4o"

    def test_check_connection_failure(self, _azure_env):
        from src.framework.azure_llm import AzureLLMProvider

        provider = AzureLLMProvider()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Connection refused")
        provider._client = mock_client

        result = provider.check_connection()
        assert result["connected"] is False
        assert "Connection refused" in result["error"]


# ---------------------------------------------------------------------------
# Tests — module helpers
# ---------------------------------------------------------------------------


class TestModuleHelpers:
    def test_azure_available_true(self, _azure_env):
        from src.framework.azure_llm import azure_available
        assert azure_available() is True

    def test_azure_available_false(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)
        from src.framework.azure_llm import azure_available
        assert azure_available() is False

    def test_get_azure_llm_singleton(self, _azure_env):
        from src.framework.azure_llm import get_azure_llm
        p1 = get_azure_llm()
        p2 = get_azure_llm()
        assert p1 is p2
