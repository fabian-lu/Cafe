"""Per-stage reporting + pipeline-topology view for composed (Mode B) systems.

The composed system itself is a :class:`~cafe.techniques.pipe.Pipeline` (which acts as
the Study ``system`` and carries the per-stage **trace** + total **cost** in each answer's
metadata). This module reads those traces: :func:`stage_report` aggregates per-stage
time/cost, and :func:`pipeline` renders the observed stage order.
"""

from __future__ import annotations

from typing import Any


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


class PipelineView:
    """The composed system's stages in execution order, each with its technique levels.

    CAFE doesn't hard-code your topology — it *observes* it from a run's trace (the order
    in which ``ctx.run`` fired). ``str(view)`` renders it as
    ``retrieve → rerank → generate`` with the levels underneath — handy for the notebook and
    (later) the web app's pipeline diagram. Get one via :func:`cafe.pipeline`.
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
        return f"PipelineView({' → '.join(self.order) or 'empty'})"


def pipeline(source: Any) -> PipelineView:
    """Derive the :class:`PipelineView` (stage order + levels) from a run.

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
            return PipelineView([])
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

    # Levels: prefer the study's declared factor levels for the stage, else the pipeline's
    # own techniques (study.system is a Pipeline), else whatever was observed.
    factor_levels: dict[str, list[Any]] = {}
    pipe = study.system if study is not None else None
    if study is not None:
        for f in study.factors:
            factor_levels[f.name] = list(f.levels)

    stages_out = []
    for st in order:
        pipe_names = pipe.names_for(st) if hasattr(pipe, "names_for") else []
        levels = factor_levels.get(st) or pipe_names or observed.get(st, [])
        stages_out.append({"stage": st, "levels": levels, "techniques_observed": observed.get(st, [])})
    return PipelineView(stages_out)
