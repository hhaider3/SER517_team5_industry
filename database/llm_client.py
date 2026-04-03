"""
llm_client.py
Centralised LLM client for the Adaptive Virtual Assistant.

Supports two providers controlled by the ``LLM_PROVIDER`` env-var:

    LLM_PROVIDER=ollama   →  free local Ollama server (default)
    LLM_PROVIDER=openai   →  paid OpenAI API

Every service that needs an LLM should import from here instead of
instantiating its own ``OpenAI(...)`` client.

Public API
----------
    get_client()            → openai.OpenAI instance (works for both providers)
    get_model()             → model name for heavy tasks
    get_fast_model()        → model name for light/fast tasks
    chat_completion(msgs)   → shortcut that calls chat.completions.create
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower().strip()

# Ollama defaults
_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
_OLLAMA_FAST_MODEL = os.environ.get("OLLAMA_FAST_MODEL", _OLLAMA_MODEL)

# OpenAI defaults
_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-2025-04-14")
_OPENAI_FAST_MODEL = os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Return a shared OpenAI-compatible client (lazy singleton)."""
    global _client
    if _client is not None:
        return _client

    if LLM_PROVIDER == "openai":
        _client = OpenAI(api_key=Config.OPENAI_API_KEY)
        logger.info("LLM client: OpenAI (model=%s)", _OPENAI_MODEL)
    else:
        # Ollama exposes an OpenAI-compatible endpoint; no real API key needed
        _client = OpenAI(
            base_url=_OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama ignores this but the library requires it
        )
        logger.info("LLM client: Ollama @ %s (model=%s)", _OLLAMA_BASE_URL, _OLLAMA_MODEL)

    return _client


def get_model() -> str:
    """Return the model name for heavy / high-quality tasks."""
    if LLM_PROVIDER == "openai":
        return _OPENAI_MODEL
    return _OLLAMA_MODEL


def get_fast_model() -> str:
    """Return the model name for lighter / faster tasks."""
    if LLM_PROVIDER == "openai":
        return _OPENAI_FAST_MODEL
    return _OLLAMA_FAST_MODEL


def is_openai() -> bool:
    """True when using the real OpenAI API (for features like Responses API)."""
    return LLM_PROVIDER == "openai"


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def chat_completion(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs: Any,
) -> str:
    """Run a chat completion and return the assistant text.

    Falls back to an error string rather than raising so callers stay robust.
    """
    model = model or get_model()
    try:
        resp = get_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.error("LLM chat_completion failed (model=%s): %s", model, exc)
        return ""