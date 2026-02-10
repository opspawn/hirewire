"""Azure OpenAI LLM Provider for AgentOS.

Provides an LLM interface backed by Azure OpenAI Service, using the
deployment configured via environment variables. Designed to be the default
provider when running on Azure infrastructure.

Environment variables:
    AZURE_OPENAI_ENDPOINT  — Azure OpenAI resource endpoint
    AZURE_OPENAI_KEY       — API key for the resource
    AZURE_OPENAI_DEPLOYMENT — Model deployment name (default: gpt-4o)
"""

from __future__ import annotations

import os
from typing import Any


class AzureLLMProvider:
    """LLM provider backed by Azure OpenAI Service.

    Uses the ``openai`` Python SDK configured for Azure endpoints.
    Supports chat completion and simple text generation.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment: str | None = None,
        api_version: str = "2024-12-01-preview",
    ) -> None:
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY", "")
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.api_version = api_version
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-initialise the Azure OpenAI client."""
        if self._client is None:
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat completion request to Azure OpenAI.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            Dict with ``content``, ``role``, ``model``, ``usage``, and
            ``finish_reason`` keys.
        """
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        choice = response.choices[0]
        return {
            "content": choice.message.content or "",
            "role": choice.message.role,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else {},
            "finish_reason": choice.finish_reason,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """Generate a text response from a simple prompt.

        Convenience wrapper around :meth:`chat_completion` that accepts a
        plain string and returns a plain string.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        result = self.chat_completion(
            messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        return result["content"]

    # ------------------------------------------------------------------
    # Connectivity check
    # ------------------------------------------------------------------

    def check_connection(self) -> dict[str, Any]:
        """Verify connectivity to the Azure OpenAI endpoint.

        Returns:
            Dict with ``connected`` (bool) and optional ``error`` (str).
        """
        try:
            result = self.chat_completion(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return {"connected": True, "model": result.get("model", self.deployment)}
        except Exception as exc:
            return {"connected": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_provider: AzureLLMProvider | None = None


def get_azure_llm(**kwargs: Any) -> AzureLLMProvider:
    """Return a cached :class:`AzureLLMProvider` singleton."""
    global _provider
    if _provider is None:
        _provider = AzureLLMProvider(**kwargs)
    return _provider


def azure_available() -> bool:
    """Return *True* if Azure OpenAI environment variables are configured."""
    return bool(
        os.environ.get("AZURE_OPENAI_ENDPOINT")
        and os.environ.get("AZURE_OPENAI_KEY")
    )
