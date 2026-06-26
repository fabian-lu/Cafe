"""Turn a composed system into a Study ``system`` the existing engine can run.

``composed(fn)`` wraps your ``fn(config, item, ctx)`` into the plain
``run(config, item) -> output`` contract. The returned answer carries the
per-stage **trace** and total **cost** in its metadata, so the same execution /
judge / statistics stack works unchanged — Mode B is just a more transparent
system under test.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from cafe.study import Factor
from cafe.techniques import registry
from cafe.techniques.context import Context

ComposedFn = Callable[[dict, Any, Context], Awaitable[Any]]


def composed(fn: ComposedFn) -> Callable[[dict, Any], Awaitable[dict]]:
    """Adapt ``fn(config, item, ctx)`` into a Study ``system``."""

    async def system(config: dict[str, Any], item: Any) -> dict[str, Any]:
        ctx = Context(config)
        result = await fn(config, item, ctx)
        output = result["output"] if isinstance(result, dict) and "output" in result else result
        return {
            "output": output,
            "cost_usd": round(ctx.total_cost, 6),
            "tokens": ctx.total_tokens,
            "trace": ctx.trace,
        }

    system.__name__ = getattr(fn, "__name__", "composed_system")
    return system


def technique_factor(stage: str, names: list[str] | None = None, **factor_kwargs: Any) -> Factor:
    """A categorical factor for a stage whose levels are its registered techniques.

    ``cafe.technique_factor("retriever")`` → ``Factor("retriever", ["bm25", "dense"])``.
    """
    levels = names if names is not None else registry.names_for(stage)
    if not levels:
        raise ValueError(f"no techniques registered for stage {stage!r}")
    return Factor(stage, levels, **factor_kwargs)


def stage_report(results: Any) -> list[dict[str, Any]]:
    """Aggregate the traces in a Results object into per-stage timing + cost.

    Returns one row per (stage, technique): mean elapsed, mean cost, count — the
    basis for per-stage statistics ("which stage is slowest / most expensive").
    """
    from collections import defaultdict

    agg: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"elapsed": 0.0, "cost": 0.0, "tokens": 0.0, "n": 0}
    )
    for obs in getattr(results, "observations", []):
        for step in (obs.metadata or {}).get("trace", []):
            if step.get("cached"):
                continue
            a = agg[(step["stage"], step["technique"])]
            a["elapsed"] += step.get("elapsed_s", 0.0)
            a["cost"] += step.get("cost_usd", 0.0)
            a["tokens"] += step.get("tokens", 0)
            a["n"] += 1
    rows = []
    for (stage, tech), a in agg.items():
        n = a["n"] or 1
        rows.append({
            "stage": stage,
            "technique": tech,
            "calls": a["n"],
            "mean_elapsed_s": round(a["elapsed"] / n, 4),
            "mean_cost_usd": round(a["cost"] / n, 6),
            "mean_tokens": round(a["tokens"] / n, 1),
        })
    return sorted(rows, key=lambda r: r["mean_elapsed_s"], reverse=True)
