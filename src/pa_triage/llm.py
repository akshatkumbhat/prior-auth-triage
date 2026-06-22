"""Thin LLM client factory.

The rest of the codebase only ever touches :func:`get_llm`, so the provider
is swappable behind a single seam. Provider SDKs are imported lazily so that,
for example, running with Ollama does not require the Gemini package to be
importable (and tests can mock this factory without importing either).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pa_triage.config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.language_models.chat_models import BaseChatModel


class LLMConfigError(RuntimeError):
    """Raised when the selected provider is misconfigured (e.g. missing key)."""


def get_llm(settings: Settings | None = None) -> "BaseChatModel":
    """Return a chat model for the configured provider.

    Parameters
    ----------
    settings:
        Optional override. Defaults to the cached :func:`get_settings`.

    The returned object is a LangChain ``BaseChatModel`` and supports
    ``.with_structured_output(PydanticModel)``, which is how agents obtain
    validated structured outputs.
    """
    settings = settings or get_settings()

    if settings.llm_provider == "gemini":
        if not settings.google_api_key:
            raise LLMConfigError(
                "LLM_PROVIDER=gemini but GOOGLE_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/app/apikey "
                "or set LLM_PROVIDER=ollama to run locally."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=settings.llm_temperature,
            # Pin to the Developer API (API-key) path. In managed containers
            # (e.g. Hugging Face Spaces) ambient Google env vars can otherwise
            # flip the client into Vertex AI mode, which authenticates via OAuth
            # and rejects AI Studio API keys with 401 ACCESS_TOKEN_TYPE_UNSUPPORTED.
            vertexai=False,
        )

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )

    raise LLMConfigError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
