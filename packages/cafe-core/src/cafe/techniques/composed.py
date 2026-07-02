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


_NONE_UNSET = object()


def technique_factor(
    stage: str,
    names: list[str] | None = None,
    *,
    none: Any = _NONE_UNSET,
    none_name: str = "none",
    **factor_kwargs: Any,
) -> Factor:
    """A categorical factor for a stage whose levels are its registered techniques.

    ``cafe.technique_factor("retriever")`` → ``Factor("retriever", ["bm25", "dense"])``.

    Pass ``none=`` to add a "skip this stage" level without writing a no-op technique —
    the honest "does this stage help at all?" contrast:

    - ``none="chunks"`` → the skip level returns the ``chunks`` input unchanged
      (a *pass-through* stage, e.g. a reranker turned off);
    - ``none=None``     → the skip level contributes nothing / returns ``None``
      (an *additive* stage, e.g. web-search turned off).

    ``none_name`` sets the level's label (default ``"none"``).
    """
    levels = list(names if names is not None else registry.names_for(stage))
    if none is not _NONE_UNSET:
        returns = None if none is None else str(none)
        registry.register_passthrough(stage, none_name, returns)
        if none_name not in levels:
            levels.append(none_name)
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


class Pipeline:
    """The composed system's stages in execution order, each with its technique levels.

    CAFE doesn't hard-code your topology — it *observes* it from a run's trace (the order
    in which ``ctx.run`` fired). ``str(pipeline)`` renders it as
    ``retriever → rerank → generator`` with the levels underneath — handy for the
    notebook and the web app's pipeline diagram.
    """

    def __init__(self, stages: list[dict[str, Any]]) -> None:
        self.stages = stages  # [{stage, levels: [..], techniques_observed: [..]}]

    @property
    def order(self) -> list[str]:
        return [s["stage"] for s in self.stages]

    def show(self) -> str:
        if not self.stages:
            return "pipeline: (no stages observed — run the study first)"
        arrow = "  →  ".join(s["stage"] for s in self.stages)
        lines = ["pipeline:  " + arrow, ""]
        for s in self.stages:
            levels = ", ".join(map(str, s["levels"])) or "—"
            lines.append(f"  {s['stage']:<14} levels: {levels}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:  # pragma: no cover - notebook display
        chips = "  <b>→</b>  ".join(
            f"<code>{s['stage']}</code>" for s in self.stages
        )
        rows = "".join(
            f"<tr><td><b>{s['stage']}</b></td><td>{', '.join(map(str, s['levels'])) or '—'}</td></tr>"
            for s in self.stages
        )
        return f"<div>{chips}</div><table><tr><th>stage</th><th>levels</th></tr>{rows}</table>"

    def __repr__(self) -> str:
        return f"Pipeline({' → '.join(self.order) or 'empty'})"


def pipeline(source: Any) -> Pipeline:
    """Derive the :class:`Pipeline` (stage order + levels) from a run.

    ``source`` may be an :class:`~cafe.evaluation.Evaluation`, a ``Results``, or a
    ``Study``. Given a study, CAFE runs a **single configuration once** (cheap) to observe
    the topology — the order comes from that trace; the levels come from the study's
    factors. Given an already-run result, no new run happens.
    """
    from cafe.study import Study

    study = source if isinstance(source, Study) else None
    results = getattr(source, "answers", source)  # Evaluation.answers, or a Results

    traces: list[list[dict[str, Any]]]
    if study is not None:
        # One run (first config × first item) is enough to see the order — no full smoke.
        from cafe.design import generate as _generate
        from cafe.system import as_system, normalize_output

        configs = _generate(study)
        item = study.dataset[0] if study.dataset else None
        if not configs or item is None:
            return Pipeline([])
        raw = study._run_blocking(lambda: as_system(study.system).run(dict(configs[0]), item))
        _, meta = normalize_output(raw)
        traces = [meta.get("trace", [])]
    else:
        traces = [(obs.metadata or {}).get("trace", []) for obs in getattr(results, "observations", [])]

    # Observed order: first appearance of each stage across the traces.
    order: list[str] = []
    observed: dict[str, list[str]] = {}
    for trace in traces:
        for step in trace:
            st = step["stage"]
            if st not in order:
                order.append(st)
                observed[st] = []
            tech = step["technique"]
            if tech not in observed[st]:
                observed[st].append(tech)

    # Levels: prefer the study's declared factor levels for the stage, else the registry.
    factor_levels: dict[str, list[Any]] = {}
    if study is not None:
        for f in study.factors:
            factor_levels[f.name] = list(f.levels)

    stages_out = []
    for st in order:
        levels = factor_levels.get(st) or registry.names_for(st) or observed.get(st, [])
        stages_out.append({"stage": st, "levels": levels, "techniques_observed": observed.get(st, [])})
    return Pipeline(stages_out)
