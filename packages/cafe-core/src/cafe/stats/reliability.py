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

    Accepts a list of dicts, a pandas DataFrame, or a **path to a filled-in CSV**
    (e.g. the sheet from :func:`answer_sheet`). Rows with a blank score are skipped, so a
    partially-rated sheet is fine; a row whose score can't be read as a number is skipped
    with a warning (naming the row) rather than aborting the whole import — a stray header
    row or a typo like ``"5?"`` in one cell shouldn't lose the sheet.
    """
    if isinstance(records, str):  # a CSV path (a filled-in answer sheet)
        import pandas as pd

        records = pd.read_csv(records)
    if hasattr(records, "to_dict"):  # a DataFrame
        records = records.to_dict("records")
    out = []
    for i, rec in enumerate(records):
        missing = {"answer_id", "rater", "score"} - set(rec)
        if missing:
            raise ValueError(f"human rating row {i} is missing {sorted(missing)}: {rec}")
        score = rec["score"]
        if score is None or score == "" or (isinstance(score, float) and score != score):
            continue  # unrated row (blank in the sheet)
        try:
            fscore = float(score)
        except (ValueError, TypeError):
            import warnings

            warnings.warn(
                f"human rating row {i} has a non-numeric score {score!r} "
                f"(answer_id={rec.get('answer_id')!r}, rater={rec.get('rater')!r}) — skipping it.",
                stacklevel=2,
            )
            continue
        out.append({
            "answer_id": rec["answer_id"],
            "rater": str(rec["rater"]),
            "score": int(fscore) if fscore.is_integer() else fscore,
        })
    return HumanRatings(records=out)


def answer_sheet(
    evaluation: "Evaluation",
    path: str | None = None,
    *,
    raters: tuple[str, ...] = ("expert_1",),
    questions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """A rating sheet: one row per (answer × rater) with a blank ``score`` to fill in.

    Give it to your experts to rate. Pass ``path="sheet.csv"`` to **write a CSV** they can
    open in Excel/Sheets, fill the ``score`` column, and save; load it back with
    ``cafe.human_ratings("sheet.csv")``. ``raters`` names the columns of experts (one row
    each). ``questions`` maps ``input_id`` → question text (defaults to the evaluation's).
    """
    from cafe.execution.results import config_label

    questions = questions if questions is not None else getattr(evaluation, "questions", {}) or {}
    references = getattr(evaluation, "references", {}) or {}
    rows = []
    for o in evaluation.answers.observations:
        if not o.ok:
            continue
        for rater in raters:
            # Column order = what a human reads left→right, then fills `score`:
            # question, the gold reference (same one the judge saw), the answer, then score.
            rows.append({
                "answer_id": o.key(),
                "rater": rater,
                "question": questions.get(o.input_id, ""),
                "reference": references.get(o.input_id, ""),
                "output": o.output,
                "score": "",
                "config": config_label(o.config),
            })
    if path is not None:
        import os

        import pandas as pd

        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)  # so "data/sheet.csv" works even if data/ is new
        pd.DataFrame(rows).to_csv(path, index=False)
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
        a0 = "n/a" if self.alpha != self.alpha else f"{self.alpha:.3f}"
        undefined = self.n_units < 2 or self.alpha != self.alpha
        interp = "undefined" if undefined else self.interpret(self.alpha)
        lines = [
            f"inter-rater reliability — Krippendorff's α ({self.metric})",
            f"  raters ({len(self.raters)}): {', '.join(self.raters)}"
            f"      answers rated by ≥2: {self.n_units}",
            "",
            f"  overall α = {a0}   ({interp})",
        ]
        if undefined:
            lines.append(
                f"  note: α needs ≥2 answers each rated by at least two raters "
                f"(got {self.n_units}); the raters here don't share enough scored answers."
            )
        if len(self.pairwise) > 1:  # only interesting with ≥3 raters
            lines.append("")
            lines.append("  pairwise:")
            w = max((len(p["a"]) for p in self.pairwise), default=6)
            for p in self.pairwise:
                a = p["alpha"]
                astr = "n/a" if a != a else f"{a:+.3f}"
                lines.append(f"    {p['a']:>{w}} ↔ {p['b']:<{w}}   α = {astr}   (n={p['n_common']})")
        lines.append("")
        lines.append("  α ≥ 0.80 reliable · 0.667–0.80 tentative · < 0.667 unreliable")
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:  # pragma: no cover - notebook display
        return f"<pre>{self.show()}</pre>"

    def __repr__(self) -> str:
        return f"Reliability(α={self.alpha:.3f} [{self.interpret(self.alpha)}], {len(self.raters)} raters)"


def _judge_scores(evaluation: "Evaluation", label: str) -> dict[Any, Any]:
    """Collapse the judge's verdicts to one score per answer (mean over judge reps).

    For an ordinal/binary rubric the mean is rounded back to an integer scale point
    (round-half-up); for a **numeric** rubric the mean is kept as-is — rounding 7.5→8 would
    corrupt the interval metric that α uses for numeric scales."""
    from cafe.judging.rubric import ScaleType

    by_answer: dict[Any, list[float]] = defaultdict(list)
    ratings = getattr(evaluation, "ratings", None)
    if ratings is not None:
        for r in ratings.items:
            if r.value_numeric is not None:
                by_answer[r.obs_key].append(float(r.value_numeric))
    scale = getattr(getattr(ratings, "rubric", None), "scale_type", None)
    if scale == ScaleType.numeric:
        return {u: statistics.fmean(v) for u, v in by_answer.items()}
    return {u: int(statistics.fmean(v) + 0.5) for u, v in by_answer.items()}  # round-half-up


def _metric_for(evaluation: "Evaluation" | None) -> str:
    """Pick the disagreement metric from the rubric's scale: ordinal→ordinal,
    numeric→interval, binary→nominal (so 1-vs-5 counts more than 1-vs-2 on an ordinal
    scale, but categories are all-or-nothing)."""
    scale = getattr(getattr(getattr(evaluation, "ratings", None), "rubric", None), "scale_type", None)
    return {"ordinal": "ordinal", "numeric": "interval", "binary": "nominal"}.get(
        getattr(scale, "value", None), "ordinal"
    )


def _scores_of(source: Any, judge_label: str = "judge") -> dict[Any, Any]:
    """A ``{answer_id: score}`` map from a rater source — an ``Evaluation`` (its judge's
    per-answer scores) or an already-built dict."""
    if isinstance(source, dict):
        return source
    if getattr(source, "answers", None) is not None:  # an Evaluation
        return _judge_scores(source, judge_label)
    raise TypeError(f"a rater must be an Evaluation or a {{answer_id: score}} dict; got {type(source).__name__}")


def reliability(
    evaluation: "Evaluation" | None = None,
    human: Any = None,
    *,
    raters: dict[str, Any] | None = None,
    table: dict[str, dict[Any, Any]] | None = None,
    metric: str | None = None,
    judge_label: str = "judge",
) -> Reliability:
    """Inter-rater reliability across judges and/or human raters.

    Common uses::

        cafe.reliability(evaluation, human=expert_scores)   # judge ↔ humans
        cafe.reliability(raters={"120b": result, "20b": result.rejudge(judge_20b)})  # judge ↔ judge
        cafe.reliability(human=expert_scores)               # humans ↔ each other

    ``raters`` maps a display name to a rater source — an ``Evaluation`` (its judge's
    scores) or a ``{answer_id: score}`` dict. ``human`` is a :class:`HumanRatings` or rows
    for :func:`human_ratings`. The disagreement ``metric`` defaults to the rubric's scale.
    """
    if metric is None:
        metric = _metric_for(evaluation or next((s for s in (raters or {}).values()
                                                 if getattr(s, "answers", None) is not None), None))
    if table is None:
        table = {}
        if raters:
            for name, src in raters.items():
                table[name] = _scores_of(src, judge_label)
        if evaluation is not None and not raters:
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


# ── Use human ratings as the ratings for the full stats ─────────────────────────────

def ratings_from_human(evaluation: "Evaluation", human: Any, *, rubric: Any = None):
    """Turn human scores into a :class:`~cafe.judging.ratings.Ratings`, so the whole stats
    stack (``attribute`` / ``fit_effects`` / ``fit_clmm`` / ``report``) runs on **humans**
    instead of the judge. Each rater's score becomes a verdict; several raters on one
    answer act like judge replications (averaged before the factor models)."""
    from cafe.judging.ratings import Rating, Ratings

    hr = human if isinstance(human, HumanRatings) else human_ratings(human)
    by_key = {o.key(): o for o in evaluation.answers.observations}
    if rubric is None:
        rubric = getattr(getattr(evaluation, "ratings", None), "rubric", None)
    if rubric is None:
        raise ValueError("no rubric available (this evaluation has no judge ratings) — pass rubric=")

    grouped: dict[Any, list[tuple[str, Any]]] = defaultdict(list)
    for rec in hr.records:
        grouped[rec["answer_id"]].append((rec["rater"], rec["score"]))

    items = []
    for aid, rs in grouped.items():
        obs = by_key.get(aid)
        if obs is None:
            continue
        for jr, (rater, score) in enumerate(sorted(rs)):
            items.append(Rating(
                obs_key=aid, config=dict(obs.config), input_id=obs.input_id, rep=obs.rep,
                judge_rep=jr, value=score, value_numeric=score, reasoning=f"human:{rater}",
            ))
    return Ratings(rubric=rubric, judge_model="human",
                   factors=list(evaluation.answers.factors), items=items)


def human_evaluation(evaluation: "Evaluation", human: Any, *, rubric: Any = None):
    """An :class:`~cafe.evaluation.Evaluation` backed by **human** ratings — so
    ``.report()`` / ``.plot()`` / ``.effects`` / ``.clmm`` all describe what the *humans*
    found, letting you compare it against the judge-backed evaluation side by side."""
    from cafe.evaluation import Evaluation
    from cafe.stats.descriptive import attribute

    rt = ratings_from_human(evaluation, human, rubric=rubric)
    return Evaluation(
        study_name=f"{evaluation.study_name} (human-rated)",
        answers=evaluation.answers, ratings=rt, attribution=attribute(rt),
        questions=getattr(evaluation, "questions", {}), references=getattr(evaluation, "references", {}),
    )
