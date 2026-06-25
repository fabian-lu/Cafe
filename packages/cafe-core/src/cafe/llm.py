"""LLM calls via LiteLLM — one interface over every provider.

We don't ship a bespoke client. :func:`complete` is a thin async wrapper over
``litellm.acompletion`` so a model string selects the provider:

- ``"gpt-4o"``                    → OpenAI        (needs ``OPENAI_API_KEY``)
- ``"anthropic/claude-..."``      → Anthropic     (needs ``ANTHROPIC_API_KEY``)
- ``"ollama_cloud/gpt-oss:120b"`` → Ollama Cloud  (needs ``OLLAMA_API_KEY``)
- ``"ollama/llama3"``             → a local Ollama daemon (no key needed)
- vLLM, Groq, Together, … — anything LiteLLM supports, by its model string

Keys are read from the environment / nearest ``.env``. ``litellm`` is imported
lazily and only required if you actually call an LLM (``pip install
'cafe-core[llm]'``). If you need something LiteLLM doesn't cover, plug in your own
:class:`cafe.Judge` instead — there is no client abstraction to implement.
"""

from __future__ import annotations

import os
from typing import Any

from cafe._env import load_env


class LLMError(RuntimeError):
    """Raised when an LLM call fails (auth, network, provider error, empty payload)."""


def _route(model: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map a ``model`` string to LiteLLM call kwargs.

    Almost everything is passed straight to LiteLLM. The one special case is
    **Ollama Cloud**: LiteLLM's native ``ollama/`` provider only talks to a *local*
    daemon (``localhost``, no auth), so to keep both possible we use an explicit
    prefix:

    - ``"ollama/<model>"``        → local daemon, handled entirely by LiteLLM.
    - ``"ollama_cloud/<model>"``  → Ollama Cloud (``ollama.com``); routed here to its
      OpenAI-compatible endpoint with ``OLLAMA_API_KEY`` (and optional ``OLLAMA_HOST``).
    """
    out = {"model": model, **kwargs}
    if model.startswith("ollama_cloud/"):
        tag = model.split("/", 1)[1]
        key = os.environ.get("OLLAMA_API_KEY")
        if not key:
            raise LLMError(
                "model 'ollama_cloud/...' needs OLLAMA_API_KEY (set it in .env). "
                "For a local Ollama daemon use 'ollama/<model>' instead."
            )
        host = (os.environ.get("OLLAMA_HOST") or "https://ollama.com").rstrip("/")
        out.update(model=f"openai/{tag}", api_base=f"{host}/v1", api_key=key)
    return out


async def complete(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    timeout: float = 120.0,
    **kwargs: Any,
) -> str:
    """Run one chat completion and return the assistant's text content."""
    load_env()
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover
        raise LLMError(
            "LLM calls need litellm; install with: pip install 'cafe-core[llm]'"
        ) from exc

    litellm.suppress_debug_info = True
    call = _route(model, dict(messages=messages, temperature=temperature, timeout=timeout, **kwargs))
    try:
        resp = await litellm.acompletion(**call)
    except Exception as exc:  # noqa: BLE001 — surface any provider error uniformly
        raise LLMError(f"{type(exc).__name__}: {exc}") from exc
    try:
        return resp.choices[0].message.content or ""
    except (AttributeError, IndexError) as exc:
        raise LLMError(f"unexpected completion payload: {resp!r}") from exc
