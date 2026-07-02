"""Judge stability: how much a judge disagrees *with itself* across repetitions.

This is the judge-side analogue of the system's ``replications``. When a study sets
``judge_replications > 1`` (or you ``rejudge(..., repetitions=k)``), every answer is
scored several times; the spread of those scores measures the judge's own
run-to-run nondeterminism, separately from the system's. A judge that wanders a lot
is a reliability red flag — pair this with inter-rater reliability (:mod:`reliability`).

Requires the ``stats`` extra only for the DataFrame view; the numbers are pure Python.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from cafe.judging import Ratings


@dataclass
class JudgeStability:
    """Per-answer spread of a judge's repeated verdicts, plus a summary."""

    judge_model: str
    n_answers: int = 0          # answers with >= 2 usable verdicts (spread is defined)
    judge_reps: int = 0         # max repetitions seen on any answer
    mean_sd: float | None = None    # mean per-answer std dev (the headline "judge noise")
    max_sd: float | None = None
    unanimous_frac: float | None = None  # share of answers scored identically every time
    per_answer: list[dict[str, Any]] = field(default_factory=list)

    def show(self) -> str:
        head = f"judge stability — {self.judge_model}: {self.n_answers} answer(s) × up to {self.judge_reps} reps"
        if not self.per_answer:
            return head + "\n  (need judge_replications ≥ 2 — nothing repeated to compare)"
        summary = (
            f"  mean sd {self.mean_sd:.2f} · max sd {self.max_sd:.2f} · "
            f"unanimous {self.unanimous_frac:.0%}   (higher sd = the judge wanders more)"
        )
        lines = [head, summary, "", "per answer:"]
        for r in sorted(self.per_answer, key=lambda d: d["sd"], reverse=True):
            lines.append(f"  {r['input_id']:<28} verdicts={r['verdicts']}  sd={r['sd']:.2f}")
        return "\n".join(lines)

    @property
    def df(self):
        """A pandas DataFrame, one row per answer (needs the ``stats`` extra)."""
        import pandas as pd

        return pd.DataFrame(self.per_answer)

    def __repr__(self) -> str:
        if self.mean_sd is None:
            return f"JudgeStability({self.judge_model}: no repetitions)"
        return f"JudgeStability({self.judge_model}: mean sd={self.mean_sd:.2f}, {self.n_answers} answers)"


def judge_stability(ratings: Ratings) -> JudgeStability:
    """Measure how consistently a judge scores the **same answer** across its repetitions.

    Groups verdicts by answer (``obs_key``) and, for each answer with ≥2 usable
    verdicts, computes the population std dev and the value list. Returns a
    :class:`JudgeStability` with per-answer rows and a summary (mean/max sd, and the
    fraction of answers the judge scored identically every time). Meaningful only when
    the judge scored answers more than once (``judge_replications`` / ``repetitions``).
    """
    result = JudgeStability(judge_model=getattr(ratings, "judge_model", "judge"))

    by_answer: dict[str, dict[str, Any]] = {}
    for r in ratings.items:
        if not r.ok:
            continue
        slot = by_answer.setdefault(
            r.obs_key, {"input_id": r.input_id, "config": dict(r.config), "reps": []}
        )
        slot["reps"].append((r.judge_rep, r.value_numeric))

    per_answer = []
    for slot in by_answer.values():
        verdicts = [v for _, v in sorted(slot["reps"])]
        result.judge_reps = max(result.judge_reps, len(verdicts))
        if len(verdicts) < 2:
            continue  # a single verdict has no spread
        sd = statistics.pstdev(verdicts)
        per_answer.append({
            "input_id": slot["input_id"],
            "config": slot["config"],
            "verdicts": verdicts,
            "sd": sd,
            "range": max(verdicts) - min(verdicts),
        })

    result.per_answer = per_answer
    result.n_answers = len(per_answer)
    if per_answer:
        sds = [r["sd"] for r in per_answer]
        result.mean_sd = statistics.fmean(sds)
        result.max_sd = max(sds)
        result.unanimous_frac = sum(1 for s in sds if s == 0) / len(sds)
    return result
