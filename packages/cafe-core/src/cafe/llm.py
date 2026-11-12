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


# A sink that, when set, collects per-call usage (tokens + cost, and energy when the model
# has a figure). The composed-mode Context sets this around each technique so LLM usage is
# attributed per stage automatically — the user just calls ``cafe.complete`` as normal.
# ``None`` (the default, e.g. black-box mode or judging) means usage is simply not collected.
_usage_sink: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "cafe_usage_sink", default=None
)

# User-set price overrides, model string -> {"input": $/1k, "output": $/1k}. Used when
# LiteLLM's automatic pricing is wrong or missing: a flat subscription, a negotiated
# rate, self-hosting, or a provider LiteLLM doesn't price (e.g. Ollama Cloud, whose
# calls otherwise get an unknown cost).
_model_prices: dict[str, dict[str, float]] = {}

# User-declared energy figures, model string -> Wh per 1k tokens. Strictly opt-in: CAFE
# never guesses one, so a call to a model without a figure records no energy of its own.
_model_energy: dict[str, float] = {}

# Models already warned about having no known price (see ``_emit_usage``): once each.
_unpriced_warned: set[str] = set()


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
    rates, self-hosted models, or providers LiteLLM doesn't price. A model with neither
    an override nor a LiteLLM price (e.g. every ``ollama_cloud/...`` model and many
    OpenRouter models) gets an *unknown* cost (``None``), not $0, and CAFE warns once
    per model. Pass all-None to clear the override.

    Parameters
    ----------
    model:
        The model string exactly as passed to :func:`complete`, e.g.
        ``"ollama_cloud/gpt-oss:120b"``.
    per_1k_tokens:
        One blended rate for prompt and completion tokens.
    per_1k_input, per_1k_output:
        Separate prompt / completion rates. Each takes precedence over ``per_1k_tokens``
        for its side; a side with neither rate is priced at 0.

    Raises
    ------
    ValueError
        If a rate is negative.
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


def set_model_energy(model: str, *, wh_per_1k_tokens: float | None = None) -> None:
    """Declare the energy a call to ``model`` uses, in Wh per 1,000 tokens.

    The energy counterpart of :func:`set_model_cost`. Each :func:`complete` call made inside
    a technique of a :class:`~cafe.techniques.pipe.Pipeline` is then charged
    ``tokens / 1000 * wh_per_1k_tokens`` for the model it called, so a technique with a
    ``model`` parameter, or one that calls two models, gets each call's energy from that
    call's model. A model's figure takes priority over the technique's own
    ``energy_wh_per_1k_tokens``, which then counts only the tokens of calls to models without
    a figure; the technique's fixed ``energy_wh`` is still added. Like a price, the figure
    holds for the whole process: every pipeline that calls ``model`` uses it.

    Energy stays opt-in: CAFE never guesses a figure, and with none declared anywhere no
    energy is reported. A black-box system reports its own ``energy_wh``; this registry
    does not apply to it.

    Parameters
    ----------
    model:
        The model string exactly as passed to :func:`complete` (e.g.
        ``"ollama_cloud/gpt-oss:20b"``), the same key :func:`set_model_cost` uses.
    wh_per_1k_tokens:
        Watt-hours per 1,000 tokens of a call (prompt plus completion tokens), e.g. an
        estimate from the model's active parameters and serving hardware. ``0.0`` declares
        the model's calls free of energy; ``None`` (the default) clears the figure.

    Raises
    ------
    ValueError
        If ``wh_per_1k_tokens`` is negative.
    """
    if wh_per_1k_tokens is None:
        _model_energy.pop(model, None)
        return
    if wh_per_1k_tokens < 0:
        raise ValueError(f"wh_per_1k_tokens must be non-negative; got {wh_per_1k_tokens}")
    _model_energy[model] = float(wh_per_1k_tokens)


def _energy_from_figure(model: str, tokens: int) -> float | None:
    """Energy (Wh) of one call from its model's declared figure, or ``None`` if it has none."""
    per_1k = _model_energy.get(model)
    if per_1k is None:
        return None
    return tokens / 1000 * per_1k


def _price_from_litellm(resp: Any) -> float | None:
    """LiteLLM's price for one completion, or ``None`` if it has no price for the model.

    ``litellm.completion_cost`` raises for a model missing from its price map
    (``ModelNotMappedError`` in current LiteLLM, a plain ``Exception`` in older ones), so
    any exception means "no price" — never a free call. A price LiteLLM does state is kept,
    including a genuine 0.0 (e.g. a local ``ollama/`` model).
    """
    try:
        import litellm

        cost = litellm.completion_cost(completion_response=resp)
    except Exception:  # noqa: BLE001 — no price for this model; the caller records "unknown"
        return None
    return None if cost is None else float(cost)


def _warn_unpriced(model: str) -> None:
    """Warn, once per model, that calls to ``model`` get an unknown cost."""
    if model in _unpriced_warned:
        return
    _unpriced_warned.add(model)
    warnings.warn(
        f"no price known for model {model!r} — the cost of its calls is recorded as unknown "
        "(None), not $0; tokens are still counted. Set a price with "
        f"cafe.set_model_cost({model!r}, per_1k_input=..., per_1k_output=...).",
        # _warn_unpriced ← _emit_usage ← complete ← the code that called cafe.complete
        stacklevel=4,
    )


def _emit_usage(model: str, resp: Any) -> None:
    """Record one completion's tokens + cost (and energy, if declared) to the sink.

    Cost priority: (1) a user override via :func:`set_model_cost`, else (2) LiteLLM's
    automatic pricing (a genuine 0.0 included), else (3) ``None``: unknown, not free. The
    tokens are recorded either way, and an unknown price warns once per model. Energy is
    recorded (as ``energy_wh``) only when ``model`` has a figure from
    :func:`set_model_energy`; without one the record has no ``energy_wh`` key, and the
    technique's own ``energy_wh_per_1k_tokens`` (if any) covers the call's tokens.
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
        cost = _price_from_litellm(resp)
        if cost is None:
            _warn_unpriced(model)
    record: dict[str, Any] = {"model": model, "tokens": tokens, "cost_usd": cost}
    energy = _energy_from_figure(model, tokens)
    if energy is not None:
        record["energy_wh"] = energy
    sink.append(record)


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
    """Run one chat completion and return the assistant's text content.

    Called inside a pipeline stage (``ctx.run``), the call's tokens and cost are recorded
    to that stage. The cost comes from :func:`set_model_cost` if a price is set for
    ``model``, else from LiteLLM's price tables; a model priced by neither gets an unknown
    cost (``None``), not $0.

    Parameters
    ----------
    model:
        A LiteLLM model string (``"gpt-4o"``, ``"anthropic/claude-..."``), or
        ``"ollama_cloud/<model>"`` for Ollama Cloud.
    messages:
        The chat messages, ``[{"role": ..., "content": ...}, ...]``.
    temperature, timeout:
        The sampling temperature, and the request timeout in seconds.
    **kwargs:
        Passed through to ``litellm.acompletion``.

    Returns
    -------
    str
        The reply's text (``""`` if it has none).

    Raises
    ------
    LLMError
        If the call can't be made or fails (missing key, auth, network, provider error),
        or returns an unexpected payload.

    Warns
    -----
    UserWarning
        Once per model, when a call inside a pipeline stage has no known price.
    """
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
