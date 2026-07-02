"""The analysis frame: one row per *answer*, with judge replications averaged.

Judge re-scores of the **same** answer are *technical* replicates — pure judge noise,
not new information about the system. Treating them as independent observations would be
pseudo-replication (it inflates N and understates uncertainty). So before any factor
model we collapse them to a single verdict per answer (mean over ``judge_rep``; rounded
back to an integer scale point for ordinal/binary rubrics).

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

    ``verdict`` is the mean over judge replications (rounded for ordinal/binary scales).
    NaN/unparseable verdicts are dropped first.
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
    scale = getattr(getattr(ratings, "rubric", None), "scale_type", None)
    if scale in (ScaleType.ordinal, ScaleType.binary):
        agg["verdict"] = agg["verdict"].round().astype(int)
    return agg
