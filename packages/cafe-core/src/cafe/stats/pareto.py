"""Multi-objective view: the quality–cost–latency Pareto frontier.

"Best configuration" is rarely one-dimensional — the top-quality config may be the
slowest and most expensive. :func:`pareto` aggregates each configuration's mean
**quality** (from the judge), **latency** (wall-clock), **cost** (USD), and
**tokens**, then keeps the **non-dominated** set: the configs you cannot improve on
one objective without losing on another. Everything else is strictly beaten.

Quality is maximized; cost / latency / tokens are minimized. An objective that
doesn't vary across configs (e.g. cost is 0 for a local model) is dropped from the
dominance test automatically, so the frontier reflects only what actually trades off.

Needs the ``stats`` extra (pandas); :meth:`ParetoResult.plot` needs matplotlib.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cafe.execution.results import config_label

if TYPE_CHECKING:
    from cafe.evaluation import Evaluation
    from cafe.execution.results import Results
    from cafe.judging.ratings import Ratings

# objective -> +1 if higher is better, -1 if lower is better
_DIRECTION = {"quality": +1, "cost": -1, "latency": -1, "tokens": -1}


@dataclass
class ParetoResult:
    """Per-configuration objective values and which configs are Pareto-optimal."""

    objectives: list[str]
    rows: list[dict[str, Any]] = field(default_factory=list)  # config, label, <objectives>, pareto_optimal

    @property
    def frontier(self) -> list[dict[str, Any]]:
        """The non-dominated configurations (the trade-off front)."""
        return [r for r in self.rows if r["pareto_optimal"]]

    def show(self) -> str:
        cols = self.objectives
        head = f"  {'':2}{'configuration':<34}" + "".join(f"{c:>12}" for c in cols)
        lines = [
            f"Pareto frontier over {', '.join(self.objectives)} "
            f"({len(self.frontier)} of {len(self.rows)} configs optimal):",
            "",
            head,
        ]
        for r in sorted(self.rows, key=lambda x: x.get("quality", 0), reverse=True):
            mark = "★ " if r["pareto_optimal"] else "  "
            vals = "".join(f"{r[c]:>12.4g}" for c in cols)
            lines.append(f"  {mark}{r['label']:<34}{vals}")
        lines.append("")
        lines.append("★ = Pareto-optimal (not dominated on every objective by another config)")
        return "\n".join(lines)

    def plot(self, x: str = "cost", y: str = "quality", *, ax: Any = None, annotate: bool = True):
        """Scatter all configs on two objectives, with the frontier highlighted.

        Defaults to quality (y) vs cost (x). Returns the matplotlib Axes.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4.5))

        dom = [r for r in self.rows if not r["pareto_optimal"]]
        front = sorted(self.frontier, key=lambda r: r[x])

        if dom:
            ax.scatter([r[x] for r in dom], [r[y] for r in dom],
                       c="#9aa0a6", s=45, label="dominated", zorder=2)
        ax.plot([r[x] for r in front], [r[y] for r in front],
                "-", color="#f5a623", alpha=0.6, zorder=3)
        ax.scatter([r[x] for r in front], [r[y] for r in front],
                   c="#f5a623", s=90, edgecolors="black", linewidths=0.6,
                   label="Pareto-optimal", zorder=4)

        if annotate:  # only the frontier — annotating every point is unreadable
            from cafe.execution.results import level_label

            keys = list(self.rows[0]["config"]) if self.rows else []
            varying = [k for k in keys if len({str(r["config"].get(k)) for r in self.rows}) > 1]
            for r in front:
                short = "·".join(str(level_label(r["config"][k])).split("/")[-1] for k in varying) or r["label"]
                ax.annotate(short, (r[x], r[y]), fontsize=7,
                            xytext=(4, 4), textcoords="offset points", color="#444")

        ax.set_xlabel(x + ("  (lower is better)" if _DIRECTION.get(x, 1) < 0 else ""))
        ax.set_ylabel(y + ("  (higher is better)" if _DIRECTION.get(y, 1) > 0 else ""))
        ax.set_title(f"Quality vs {x}: Pareto frontier")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)
        return ax

    def _repr_html_(self) -> str:  # pragma: no cover - notebook display
        return f"<pre>{self.show()}</pre>"

    def __repr__(self) -> str:
        return f"ParetoResult({len(self.frontier)}/{len(self.rows)} optimal over {self.objectives})"


def _dominates(a: dict, b: dict, objectives: list[str]) -> bool:
    """True if config ``a`` is at least as good as ``b`` on every objective and
    strictly better on at least one (objectives oriented by ``_DIRECTION``)."""
    at_least_as_good = True
    strictly_better = False
    for obj in objectives:
        sign = _DIRECTION[obj]
        av, bv = sign * a[obj], sign * b[obj]
        if av < bv:
            at_least_as_good = False
            break
        if av > bv:
            strictly_better = True
    return at_least_as_good and strictly_better


def _quality_by_config(ratings: "Ratings") -> tuple[dict[str, list[float]], dict[str, dict]]:
    quality: dict[str, list[float]] = defaultdict(list)
    configs: dict[str, dict] = {}
    for r in ratings.items:
        if r.value_numeric is None:
            continue
        label = config_label(r.config)
        configs[label] = dict(r.config)
        quality[label].append(float(r.value_numeric))
    return quality, configs


def _resources_by_config(answers: "Results") -> dict[str, dict[str, list[float]]]:
    res: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"latency": [], "cost": [], "tokens": []}
    )
    for o in answers.observations:
        if not o.ok:
            continue
        label = config_label(o.config)
        md = o.metadata or {}
        if o.elapsed_s is not None:
            res[label]["latency"].append(float(o.elapsed_s))
        res[label]["cost"].append(float(md.get("cost_usd", 0.0) or 0.0))
        res[label]["tokens"].append(float(md.get("tokens", 0.0) or 0.0))
    return res


def pareto(
    result: "Evaluation",
    *,
    objectives: list[str] | None = None,
) -> ParetoResult:
    """Compute the quality–cost–latency Pareto frontier over a study's configurations.

    Pass the :class:`~cafe.evaluation.Evaluation` returned by ``cafe.evaluate``.
    By default the objectives are quality plus whichever of cost / latency / tokens
    actually vary across configurations.
    """
    import statistics

    ratings = getattr(result, "ratings", None)
    answers = getattr(result, "answers", None)
    if ratings is None or answers is None:
        raise ValueError("pareto() needs an evaluated study (answers + ratings)")

    quality, configs = _quality_by_config(ratings)
    resources = _resources_by_config(answers)

    rows: list[dict[str, Any]] = []
    for label, cfg in configs.items():
        r = resources.get(label, {"latency": [], "cost": [], "tokens": []})
        rows.append({
            "config": cfg,
            "label": label,
            "quality": round(statistics.fmean(quality[label]), 4) if quality[label] else 0.0,
            "latency": round(statistics.fmean(r["latency"]), 4) if r["latency"] else 0.0,
            "cost": round(statistics.fmean(r["cost"]), 6) if r["cost"] else 0.0,
            "tokens": round(statistics.fmean(r["tokens"]), 1) if r["tokens"] else 0.0,
        })

    # Choose objectives: quality always; resource objectives only if they vary.
    if objectives is None:
        objectives = ["quality"]
        for obj in ("cost", "latency", "tokens"):
            values = {r[obj] for r in rows}
            if len(values) > 1:
                objectives.append(obj)

    for r in rows:
        r["pareto_optimal"] = not any(
            _dominates(other, r, objectives) for other in rows if other is not r
        )

    return ParetoResult(objectives=objectives, rows=rows)
