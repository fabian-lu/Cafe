"""Inter-rater reliability — is the judge actually measuring quality?

An LLM judge is only worth trusting if it agrees with humans (and with itself, and
other judges). This module computes **Krippendorff's α** — the standard
reliability coefficient that handles any number of raters, missing ratings, and
ordinal scales — between any set of raters: judge↔human, human↔human, judge↔judge.

Workflow:
  1. ``sheet = cafe.answer_sheet(evaluation)`` → one row per answer with a stable
     ``answer_id`` (plus the output to read). Hand it to your experts.
  2. Collect their scores and load them: ``cafe.human_ratings([{answer_id, rater, score}, ...])``.
  3. ``cafe.reliability(evaluation, human=...)`` → α overall and per rater pair.

α ≥ 0.80 is conventionally reliable; 0.67–0.80 tentative; below that, the judge and
humans don't agree enough to trust the judge's verdicts. Needs the ``stats`` extra.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cafe.evaluation import Evaluation

_METRICS = ("nominal", "ordinal", "interval")


# ── Krippendorff's alpha ────────────────────────────────────────────────────────

def krippendorff_alpha(table: dict[str, dict[Any, Any]], metric: str = "ordinal") -> float:
    """Krippendorff's α for ``{rater: {unit: value}}``. Missing ratings are fine.

    ``metric`` is the disagreement metric: ``"ordinal"`` (default, for rating
    scales), ``"nominal"`` (categories), or ``"interval"`` (numeric distance).
    Returns NaN if there's nothing pairable.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}; got {metric!r}")

    units: dict[Any, list[Any]] = defaultdict(list)
    for scores in table.values():
        for unit, val in scores.items():
            if val is not None:
                units[unit].append(val)
    pairable = [vs for vs in units.values() if len(vs) >= 2]
    if not pairable:
        return float("nan")

    values = sorted({v for vs in pairable for v in vs})
    idx = {v: i for i, v in enumerate(values)}
    V = len(values)

    # Coincidence matrix: each unit contributes its rating pairs, weighted 1/(m-1).
    o = [[0.0] * V for _ in range(V)]
    for vs in pairable:
        m = len(vs)
        w = 1.0 / (m - 1)
        for a in range(m):
            ia = idx[vs[a]]
            for b in range(m):
                if a != b:
                    o[ia][idx[vs[b]]] += w

    n_c = [sum(row) for row in o]
    n = sum(n_c)
    if n < 2 or V < 2:
        return 1.0  # everyone agrees on a single value → perfectly reliable

    def delta2(a: int, b: int) -> float:
        if metric == "nominal":
            return 0.0 if a == b else 1.0
        if metric == "interval":
            return (values[a] - values[b]) ** 2
        lo, hi = (a, b) if a <= b else (b, a)  # ordinal
        between = sum(n_c[g] for g in range(lo, hi + 1))
        return (between - (n_c[a] + n_c[b]) / 2.0) ** 2

    num = den = 0.0
    for a in range(V):
        for b in range(V):
            d = delta2(a, b)
            num += o[a][b] * d
            den += n_c[a] * n_c[b] * d
    if den == 0:
        return 1.0
    return 1.0 - (n - 1) * num / den


# ── Human ratings ingestion ─────────────────────────────────────────────────────

@dataclass
class HumanRatings:
    """Human (or external) scores keyed by ``answer_id`` — the human counterpart of
    judge :class:`~cafe.judging.ratings.Ratings`."""

    records: list[dict[str, Any]] = field(default_factory=list)  # {answer_id, rater, score}

    def raters(self) -> list[str]:
        return sorted({r["rater"] for r in self.records})

    def by_rater(self) -> dict[str, dict[Any, Any]]:
        out: dict[str, dict[Any, Any]] = defaultdict(dict)
        for r in self.records:
            out[r["rater"]][r["answer_id"]] = r["score"]
        return dict(out)

    def __len__(self) -> int:
        return len(self.records)


def human_ratings(records: Any) -> HumanRatings:
    """Build :class:`HumanRatings` from rows of ``{answer_id, rater, score}``.

    Accepts a list of dicts or a pandas DataFrame (with those columns).
    """
    if hasattr(records, "to_dict"):  # a DataFrame
        records = records.to_dict("records")
    out = []
    for i, rec in enumerate(records):
        missing = {"answer_id", "rater", "score"} - set(rec)
        if missing:
            raise ValueError(f"human rating row {i} is missing {sorted(missing)}: {rec}")
        score = rec["score"]
        out.append({
            "answer_id": rec["answer_id"],
            "rater": str(rec["rater"]),
            "score": None if score is None else (int(score) if float(score).is_integer() else float(score)),
        })
    return HumanRatings(records=out)


def answer_sheet(evaluation: "Evaluation", questions: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """One row per answer with a stable ``answer_id`` for humans to rate.

    Give the resulting rows (or their DataFrame) to your experts; they fill a
    ``score`` column keyed by ``answer_id``. ``questions`` optionally maps
    ``input_id`` → question text to include for context.
    """
    questions = questions or {}
    rows = []
    for o in evaluation.answers.observations:
        if not o.ok:
            continue
        from cafe.execution.results import config_label

        rows.append({
            "answer_id": o.key(),
            "config": config_label(o.config),
            "input_id": o.input_id,
            "question": questions.get(o.input_id, ""),
            "output": o.output,
        })
    return rows


# ── Reliability result + top-level entry point ──────────────────────────────────

@dataclass
class Reliability:
    """Krippendorff's α overall and for each pair of raters."""

    alpha: float
    metric: str
    raters: list[str]
    n_units: int
    pairwise: list[dict[str, Any]] = field(default_factory=list)  # {a, b, alpha, n_common}
    note: str = ""

    @staticmethod
    def interpret(alpha: float) -> str:
        if alpha != alpha:  # NaN
            return "n/a"
        if alpha >= 0.80:
            return "reliable"
        if alpha >= 0.667:
            return "tentative"
        return "unreliable"

    def show(self) -> str:
        lines = [
            f"Krippendorff's α ({self.metric}) — {len(self.raters)} raters, "
            f"{self.n_units} jointly-rated answers",
            "",
            f"  overall α = {self.alpha:.3f}   → {self.interpret(self.alpha)}",
        ]
        if self.pairwise:
            lines.append("")
            lines.append("  pairwise:")
            for p in self.pairwise:
                a = p["alpha"]
                astr = "  n/a" if a != a else f"{a:+.3f}"
                lines.append(f"     {p['a']:>12} ↔ {p['b']:<12}  α={astr}  (n={p['n_common']})")
        lines.append("")
        lines.append("  α≥0.80 reliable · 0.667–0.80 tentative · below unreliable")
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:  # pragma: no cover - notebook display
        return f"<pre>{self.show()}</pre>"

    def __repr__(self) -> str:
        return f"Reliability(α={self.alpha:.3f} [{self.interpret(self.alpha)}], {len(self.raters)} raters)"


def _judge_scores(evaluation: "Evaluation", label: str) -> dict[Any, Any]:
    """Collapse the judge's verdicts to one score per answer (mean over judge reps)."""
    by_answer: dict[Any, list[float]] = defaultdict(list)
    ratings = getattr(evaluation, "ratings", None)
    if ratings is not None:
        for r in ratings.items:
            if r.value_numeric is not None:
                by_answer[r.obs_key].append(float(r.value_numeric))
    return {u: round(statistics.fmean(v)) for u, v in by_answer.items()}


def reliability(
    evaluation: "Evaluation" | None = None,
    human: Any = None,
    *,
    table: dict[str, dict[Any, Any]] | None = None,
    metric: str = "ordinal",
    judge_label: str = "judge",
) -> Reliability:
    """Inter-rater reliability across the judge and any human raters.

    Common uses::

        cafe.reliability(evaluation, human=expert_scores)   # judge ↔ humans
        cafe.reliability(human=expert_scores)               # humans ↔ each other
        cafe.reliability(table={"judgeA": {...}, "judgeB": {...}})  # any raters

    ``human`` is a :class:`HumanRatings` or rows accepted by :func:`human_ratings`.
    """
    if table is None:
        table = {}
        if evaluation is not None:
            js = _judge_scores(evaluation, judge_label)
            if js:
                table[judge_label] = js
        if human is not None:
            hr = human if isinstance(human, HumanRatings) else human_ratings(human)
            table.update(hr.by_rater())

    if len(table) < 2:
        raise ValueError(
            "reliability needs at least two raters (e.g. the judge + one human, "
            "or two judges); got: " + (", ".join(table) or "none")
        )

    units: dict[Any, int] = defaultdict(int)
    for scores in table.values():
        for u, v in scores.items():
            if v is not None:
                units[u] += 1
    n_units = sum(1 for c in units.values() if c >= 2)

    raters = list(table)
    pairwise = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            a, b = raters[i], raters[j]
            common = sum(
                1 for u in set(table[a]) & set(table[b])
                if table[a][u] is not None and table[b][u] is not None
            )
            pairwise.append({
                "a": a, "b": b,
                "alpha": krippendorff_alpha({a: table[a], b: table[b]}, metric),
                "n_common": common,
            })

    return Reliability(
        alpha=krippendorff_alpha(table, metric),
        metric=metric,
        raters=raters,
        n_units=n_units,
        pairwise=pairwise,
    )
