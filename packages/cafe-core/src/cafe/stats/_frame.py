"""The analysis frame: one row per *answer*, with judge replications averaged.

Judge re-scores of the **same** answer are *technical* replicates — pure judge noise,
not new information about the system. Treating them as independent observations would be
pseudo-replication (it inflates N and understates uncertainty). So before any factor
model we collapse them to a single verdict per answer (mean over ``judge_rep``; for
ordinal/binary rubrics moved to the nearest level of the rubric, see
:meth:`cafe.judging.rubric.Rubric.snap`).

**System** replications are different — each is a genuinely different answer (run-to-run
nondeterminism, which CAFE exists to measure), so they keep their own rows. The grouping
key is therefore the answer: every factor + ``input_id`` + ``rep``.

This is a no-op at the default ``judge_replications=1`` (one rating per answer already).
Requires the ``stats`` extra (pandas).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cafe.judging.ratings import Ratings


def analysis_frame(ratings: "Ratings"):
    """A DataFrame with one row per answer: factors + ``input_id`` + ``rep`` + ``verdict``.

    ``verdict`` is the mean over judge replications; for ordinal/binary scales it is the
    nearest level of the rubric (a tie goes to the higher level). NaN/unparseable verdicts
    are dropped first.
    """
    import pandas as pd

    from cafe.judging.rubric import ScaleType

    df = pd.DataFrame(ratings.to_records())
    if df.empty or "verdict" not in df.columns:
        return df
    df = df.dropna(subset=["verdict"]).copy()
    if df.empty:
        return df
    df["verdict"] = df["verdict"].astype(float)

    factor_cols = [f for f in ratings.factors if f in df.columns]
    key_cols = factor_cols + [c for c in ("input_id", "rep") if c in df.columns]
    if not key_cols:
        return df  # nothing to group on — return the usable rows as they are

    agg = df.groupby(key_cols, dropna=False)["verdict"].mean().reset_index()
    rubric = getattr(ratings, "rubric", None)
    if getattr(rubric, "scale_type", None) in (ScaleType.ordinal, ScaleType.binary):
        # The nearest defined level, a tie going up: a mean of 2 on a 1/3/5 scale must not
        # become an undefined level 2, and a tied binary split (mean 0.5) is a pass.
        agg["verdict"] = agg["verdict"].map(rubric.snap).astype(int)
    return agg
