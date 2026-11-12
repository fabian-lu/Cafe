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

``ollama_cloud/`` models go to ``https://ollama.com``, or to the endpoint in
``OLLAMA_HOST`` if that is an ``https://`` URL. Any other ``OLLAMA_HOST`` (Ollama's own
tools read it as the address of an Ollama server, e.g. ``127.0.0.1:11434``) is ignored
for them, with a warning.
"""

from __future__ import annotations

import contextvars
import os
import warnings
from typing import Any

from cafe._env import load_env

# LiteLLM's background logging worker sometimes leaves a success-logging coroutine
# unawaited when the event loop tears down (e.g. the sync study.evaluate() worker thread).
# It's harmless — the completion already returned — but noisy, so silence just that message.
warnings.filterwarnings(
    "ignore",
    message=r"coroutine 'Logging\.async_success_handler' was never awaited",
    category=RuntimeWarning,
)


class LLMError(RuntimeError):
    """Raised when an LLM call fails (auth, network, provider error, empty payload)."""


# A sink that, when set, collects per-call usage (tokens + cost). The composed-mode
# Context sets this around each technique so LLM usage is attributed per stage
# automatically — the user just calls ``cafe.complete`` as normal. ``None`` (the
# default, e.g. black-box mode or judging) means usage is simply not collected.
_usage_sink: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "cafe_usage_sink", default=None
)

# User-set price overrides, model string -> {"input": $/1k, "output": $/1k}. Used when
# LiteLLM's automatic pricing is wrong or missing: a flat subscription, a negotiated
# rate, self-hosting, or a provider LiteLLM doesn't price (e.g. Ollama Cloud → $0 auto).
_model_prices: dict[str, dict[str, float]] = {}


def set_model_cost(
    model: str,
    *,
    per_1k_tokens: float | None = None,
    per_1k_input: float | None = None,
    per_1k_output: float | None = None,
) -> None:
    """Override the price CAFE uses for ``model`` (USD per 1,000 tokens).

    Use ``per_1k_tokens`` for a single blended rate, or ``per_1k_input`` /
    ``per_1k_output`` for separate prompt/completion rates. This takes priority over
    LiteLLM's automatic pricing — set it for subscriptions, negotiated/enterprise
    rates, self-hosted models, or providers LiteLLM doesn't price. Pass all-None to
    clear the override.
    """
    if per_1k_tokens is None and per_1k_input is None and per_1k_output is None:
        _model_prices.pop(model, None)
        return
    for name, val in (("per_1k_tokens", per_1k_tokens), ("per_1k_input", per_1k_input),
                      ("per_1k_output", per_1k_output)):
        if val is not None and val < 0:
            raise ValueError(f"{name} must be non-negative; got {val}")
    inp = per_1k_input if per_1k_input is not None else (per_1k_tokens or 0.0)
    out = per_1k_output if per_1k_output is not None else (per_1k_tokens or 0.0)
    _model_prices[model] = {"input": inp, "output": out}


def _price_from_override(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Cost from a user-set price, or ``None`` if this model has no override."""
    price = _model_prices.get(model)
    if price is None:
        return None
    return round(prompt_tokens / 1000 * price["input"] + completion_tokens / 1000 * price["output"], 6)


def _emit_usage(model: str, resp: Any) -> None:
    """Record one completion's tokens + cost to the sink.

    Cost priority: (1) a user override via :func:`set_model_cost`, else (2) LiteLLM's
    automatic pricing, else (3) 0.0 (tokens are still recorded, so it can be priced later).
    """
    sink = _usage_sink.get()
    if sink is None:
        return
    usage = getattr(resp, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
    tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0

    cost = _price_from_override(model, prompt_tokens, completion_tokens)
    if cost is None:
        try:
            import litellm

            cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
        except Exception:  # noqa: BLE001 — pricing unknown (e.g. local/Ollama) → cost 0
            cost = 0.0
    sink.append({"model": model, "tokens": tokens, "cost_usd": cost})


# Where ollama_cloud/ models go unless OLLAMA_HOST names another https:// endpoint.
_OLLAMA_CLOUD = "https://ollama.com"
# OLLAMA_HOST values already warned about as not used for ollama_cloud/ models: once each.
_ollama_host_warned: set[str] = set()


def _ollama_cloud_base() -> str:
    """The base URL of Ollama Cloud's OpenAI-compatible API (``<host>/v1``) for this call.

    The host is ``OLLAMA_HOST`` when that is an ``https://`` URL, else
    ``https://ollama.com``. Surrounding whitespace, trailing slashes and a trailing ``/v1``
    or ``/api`` are dropped, so ``https://ollama.com/v1`` does not become ``…/v1/v1``.

    Ollama's own tools (the ``ollama`` CLI, server and Python client) read the same variable
    as the address of an Ollama server, such as a local daemon: ``127.0.0.1:11434``,
    ``0.0.0.0``, ``http://localhost:11434``. Such a value, often exported in the shell
    (where it wins over ``.env``), is never Ollama Cloud, which is served over HTTPS
    only. So a value without ``https://`` is not used here, with a warning once per value,
    rather than sending ``ollama_cloud/`` calls and the cloud key to that server, or to a
    URL without a scheme.
    """
    raw = os.environ.get("OLLAMA_HOST") or ""
    host = raw.strip().rstrip("/")
    if not host:
        return f"{_OLLAMA_CLOUD}/v1"
    if not host.lower().startswith("https://"):
        if raw not in _ollama_host_warned:
            _ollama_host_warned.add(raw)
            warnings.warn(
                f"OLLAMA_HOST={raw!r} is not an https:// URL, so 'ollama_cloud/' models ignore "
                f"it and go to {_OLLAMA_CLOUD}. (Ollama's own tools read OLLAMA_HOST as the "
                "address of an Ollama server; reach a local one with 'ollama/<model>'.) To send "
                "'ollama_cloud/' models elsewhere, set OLLAMA_HOST to an https:// URL.",
                stacklevel=4,  # _ollama_cloud_base ← _route ← complete ← its caller
            )
        return f"{_OLLAMA_CLOUD}/v1"
    for suffix in ("/v1", "/api"):  # the API path pasted with the host: it is added below
        if host.lower().endswith(suffix):
            host = host[: -len(suffix)].rstrip("/")
            break
    return f"{host}/v1"


def _route(model: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map a ``model`` string to LiteLLM call kwargs.

    Almost everything is passed straight to LiteLLM. The one special case is
    **Ollama Cloud**: LiteLLM's native ``ollama/`` provider only talks to a *local*
    daemon (``localhost``, no auth), so to keep both possible we use an explicit
    prefix:

    - ``"ollama/<model>"``        → local daemon, handled entirely by LiteLLM.
    - ``"ollama_cloud/<model>"``  → Ollama Cloud (``ollama.com``); routed here to its
      OpenAI-compatible endpoint with ``OLLAMA_API_KEY``. ``OLLAMA_HOST`` may name
      another ``https://`` endpoint; a value in the form Ollama's own tools use for an
      Ollama server (``127.0.0.1:11434``, ``http://localhost:11434``) is not used for it
      (see :func:`_ollama_cloud_base`).
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
        out.update(model=f"openai/{tag}", api_base=_ollama_cloud_base(), api_key=key)
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
    _emit_usage(model, resp)
    try:
        return resp.choices[0].message.content or ""
    except (AttributeError, IndexError) as exc:
        raise LLMError(f"unexpected completion payload: {resp!r}") from exc
