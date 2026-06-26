"""The technique registry — the extension point for Mode B (composed pipelines).

A *technique* is a unit of your compound system (a retriever, a reranker, a model
call, a verifier), registered under a user-chosen **stage** and a **name**:

    @cafe.technique("retriever", "bm25")
    async def bm25(ctx, query, top_k=5): ...

The stage is *your* grouping, not a fixed pipeline slot — CAFE uses it to swap
techniques (the factor ``retriever`` has one level per registered name) and to do
per-stage statistics. A technique's keyword arguments *with defaults* (``top_k``)
are its tunable parameters and become parameter factors; arguments *without*
defaults (``query``) are runtime inputs you pass via ``ctx.run``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

REGISTRY: dict[tuple[str, str], "TechniqueSpec"] = {}


@dataclass
class TechniqueSpec:
    stage: str
    name: str
    fn: Callable[..., Any]
    params: dict[str, Any]      # tunable parameter -> default
    description: str = ""


def technique(stage: str, name: str, *, description: str = "") -> Callable:
    """Register ``fn`` as the technique ``name`` for ``stage``. Use as a decorator."""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)
        params = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.default is not inspect.Parameter.empty
        }
        key = (stage, name)
        if key in REGISTRY:
            raise ValueError(f"technique {stage!r}/{name!r} is already registered")
        REGISTRY[key] = TechniqueSpec(stage, name, fn, params, description)
        return fn

    return deco


def get(stage: str, name: str) -> TechniqueSpec:
    try:
        return REGISTRY[(stage, name)]
    except KeyError:
        avail = names_for(stage)
        raise KeyError(
            f"no technique {name!r} registered for stage {stage!r}; have: {avail}"
        ) from None


def names_for(stage: str) -> list[str]:
    """All technique names registered under ``stage`` (registration order)."""
    return [n for (s, n) in REGISTRY if s == stage]


def stages() -> list[str]:
    """All stages that have at least one registered technique."""
    seen: list[str] = []
    for s, _ in REGISTRY:
        if s not in seen:
            seen.append(s)
    return seen


def clear() -> None:
    """Empty the registry (mainly for tests / notebook re-runs)."""
    REGISTRY.clear()
