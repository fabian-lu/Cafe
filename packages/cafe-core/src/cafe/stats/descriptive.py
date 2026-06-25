"""Attribution statistics over judge ratings (descriptive layer).

Slice 2 ships the descriptive layer: per-configuration mean quality and per-factor
marginal means. This already answers "which configuration scored best" and "which
factor moves the needle." The inferential layers — mixed-effects model with a
question random effect, effect sizes, and the ordinal CLMM with significance — land
in a later slice; this module is where they will attach.

Requires the ``stats`` extra (pandas/numpy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cafe.judging import Ratings


@dataclass
class Attribution:
    """Descriptive attribution result."""

    n_ratings: int = 0
    n_usable: int = 0
    factors: list[str] = field(default_factory=list)
    config_means: list[dict[str, Any]] = field(default_factory=list)   # per configuration
    factor_marginals: list[dict[str, Any]] = field(default_factory=list)  # per factor level
    best_config: dict[str, Any] | None = None

    def show(self) -> str:
        lines = [
            f"ratings: {self.n_ratings} ({self.n_usable} usable)   factors: {', '.join(self.factors)}",
            "",
            "per-configuration mean quality:",
        ]
        for c in sorted(self.config_means, key=lambda r: r["mean"], reverse=True):
            cfg = "·".join(f"{k}={c['config'][k]}" for k in sorted(c["config"]))
            lines.append(f"  {c['mean']:.2f}  (n={c['n']:>2})  {cfg}")
        lines.append("")
        lines.append("per-factor marginal means:")
        cur = None
        for m in self.factor_marginals:
            if m["factor"] != cur:
                cur = m["factor"]
                lines.append(f"  {cur}:")
            lines.append(f"     {m['level']:<16} mean={m['mean']:.2f}  n={m['n']}")
        if self.best_config is not None:
            cfg = "·".join(f"{k}={self.best_config['config'][k]}" for k in sorted(self.best_config["config"]))
            lines.append("")
            lines.append(f"best configuration: {cfg}  (mean {self.best_config['mean']:.2f})")
        return "\n".join(lines)


def attribute(ratings: Ratings) -> Attribution:
    """Compute per-config and per-factor mean quality from judge ratings."""
    import pandas as pd

    df = pd.DataFrame(ratings.to_records())
    result = Attribution(n_ratings=len(df))
    if df.empty or "verdict" not in df.columns:
        return result

    df = df.dropna(subset=["verdict"]).copy()
    result.n_usable = len(df)
    if df.empty:
        return result
    df["verdict"] = df["verdict"].astype(float)

    # Factor columns are exactly the study's factor names (carried on Ratings).
    factor_cols = [f for f in ratings.factors if f in df.columns]
    result.factors = list(factor_cols)

    # Per-configuration means.
    if factor_cols:
        for keys, sub in df.groupby(factor_cols, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            config = {col: k for col, k in zip(factor_cols, keys)}
            result.config_means.append(
                {"config": config, "mean": float(sub["verdict"].mean()), "n": int(len(sub))}
            )
        result.best_config = max(result.config_means, key=lambda r: r["mean"])

    # Per-factor marginal means.
    for col in factor_cols:
        for level, sub in df.groupby(col, dropna=True):
            result.factor_marginals.append(
                {
                    "factor": col,
                    "level": str(level),
                    "mean": float(sub["verdict"].mean()),
                    "n": int(len(sub)),
                }
            )
    return result
