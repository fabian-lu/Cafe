"""Per-stage reporting + pipeline-topology view for composed (Mode B) systems.

The composed system itself is a :class:`~cafe.techniques.pipe.Pipeline` (which acts as
the Study ``system`` and carries the per-stage **trace** + total **cost** in each answer's
metadata). This module reads those traces: :func:`stage_report` aggregates per-stage
time/cost, and :func:`pipeline` renders the observed stage order.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _steps(trace: Any) -> list[dict[str, Any]]:
    """The pipeline steps in an answer's ``trace``. A black-box system may return a
    ``trace`` of its own (strings, or dicts of another shape); those entries aren't
    stages and are skipped."""
    if not isinstance(trace, (list, tuple)):
        return []
    return [s for s in trace if isinstance(s, dict)
            and isinstance(s.get("stage"), str) and isinstance(s.get("technique"), str)]


def _stage_order(traces: list[list[dict[str, Any]]]) -> list[str]:
    """Merge the stage sequences of many traces into one execution order.

    Stage ``a`` goes before ``b`` when ``a`` first ran before ``b`` in more traces than the
    other way round, so a stage that only some configurations run (a query rewrite before
    retrieval) still sits where it runs, and the order doesn't depend on which answers
    finished first. Stages the traces leave unordered (parallel branches, or orders that
    contradict each other) go by their mean position in the traces, then by name.
    """
    before: Counter[tuple[str, str]] = Counter()
    positions: dict[str, list[int]] = defaultdict(list)
    for trace in traces:
        firsts = list(dict.fromkeys(step["stage"] for step in trace))  # each stage's first run
        for i, a in enumerate(firsts):
            positions[a].append(i)
            for b in firsts[i + 1:]:
                before[a, b] += 1
    rank = {s: (sum(p) / len(p), s) for s, p in positions.items()}
    waiting = {s: {a for a in positions if before[a, s] > before[s, a]} for s in positions}
    order: list[str] = []
    while waiting:
        ready = [s for s, after in waiting.items() if not after] or list(waiting)  # or a cycle
        nxt = min(ready, key=rank.__getitem__)
        order.append(nxt)
        del waiting[nxt]
        for after in waiting.values():
            after.discard(nxt)
    return order


def stage_report(results: Any) -> list[dict[str, Any]]:
    """Aggregate the traces of a run into per-stage timing + cost.

    A repeated identical call that a run served from its stage cache (``cached: True``
    in the trace) is not counted again.

    Parameters
    ----------
    results:
        The run: a ``Results``, or an :class:`~cafe.evaluation.Evaluation` (its answers
        are used).

    Returns
    -------
    list of dict
        One row per (stage, technique): mean elapsed, mean cost, count — the basis for
        per-stage statistics ("which stage is slowest / most expensive"), slowest first.
        Empty for a system that isn't a composed pipeline: a ``trace`` such a system
        returns itself is not read as stages.
    """
    results = getattr(results, "answers", results)  # Evaluation.answers, or a Results
    agg: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"elapsed": 0.0, "cost": 0.0, "tokens": 0.0, "energy": 0.0, "energy_n": 0, "n": 0}
    )
    for obs in getattr(results, "observations", []):
        for step in _steps((obs.metadata or {}).get("trace")):
            if step.get("cached"):
                continue
            a = agg[(step["stage"], step["technique"])]
            a["elapsed"] += step.get("elapsed_s", 0.0)
            a["cost"] += step.get("cost_usd", 0.0)
            a["tokens"] += step.get("tokens", 0)
            if "energy_wh" in step:  # opt-in: only techniques that declared energy have it
                a["energy"] += step["energy_wh"]
                a["energy_n"] += 1
            a["n"] += 1
    rows = []
    for (stage, tech), a in agg.items():
        n = a["n"] or 1
        row = {
            "stage": stage,
            "technique": tech,
            "calls": a["n"],
            "mean_elapsed_s": round(a["elapsed"] / n, 4),
            "mean_cost_usd": round(a["cost"] / n, 6),
            "mean_tokens": round(a["tokens"] / n, 1),
        }
        if a["energy_n"]:
            row["mean_energy_wh"] = round(a["energy"] / a["energy_n"], 6)
        rows.append(row)
    return sorted(rows, key=lambda r: r["mean_elapsed_s"], reverse=True)


class PipelineView:
    """The composed system's stages in execution order, each with its technique levels.

    CAFE doesn't hard-code your topology — it *observes* it from a run's traces (the order
    in which ``ctx.run`` fired, merged across configurations, so a stage that only some
    configurations run still sits where it runs). ``view.show()`` renders it as
    ``retrieve → rerank → generate`` with the levels underneath — handy for the notebook and
    (later) the web app's pipeline diagram; ``str(view)`` is the one-line
    ``PipelineView(retrieve → rerank → generate)``. Get one via :func:`cafe.pipeline`.
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

    Given a study, CAFE runs a **single configuration once** (cheap) to observe the
    topology — the order comes from that trace; the levels come from the study's
    factors. Given an already-run result, no new run happens: the order merges every
    answer's trace (a stage goes after the stages it followed in most traces, so an
    optional stage that runs first in some configurations is shown first), and the
    levels are the techniques observed per stage.

    Parameters
    ----------
    source:
        An :class:`~cafe.evaluation.Evaluation`, a ``Results``, or a ``Study``.

    Returns
    -------
    PipelineView
        The stages in execution order, each with its levels. Empty for a system that
        isn't a composed pipeline: a ``trace`` such a system returns itself is not read
        as stages.
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
        traces = [_steps(meta.get("trace"))]
    else:
        traces = [_steps((obs.metadata or {}).get("trace"))
                  for obs in getattr(results, "observations", [])]

    order = _stage_order(traces)
    observed: dict[str, set[str]] = defaultdict(set)
    for trace in traces:
        for step in trace:
            observed[step["stage"]].add(step["technique"])

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
        techniques = sorted(observed[st])  # not first-seen: that order is completion order
        levels = factor_levels.get(st) or pipe_names or techniques
        stages_out.append({"stage": st, "levels": levels, "techniques_observed": techniques})
    return PipelineView(stages_out)
