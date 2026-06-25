"""Inferential statistics over judge ratings — the paper's spine.

Turns verdicts into *which factor matters, by how much, and is it real*:

- a **mixed-effects model** ``verdict ~ factors + (1 | question)`` (random intercept
  per input, since the same questions recur across configs),
- per-factor **F / p** and **partial η²** (the share of variance a factor explains),
- pairwise **Cohen's d** effect sizes with CIs.

Robust by design: falls back to one-way ANOVA when the mixed model isn't
identifiable (too few questions/levels) so it always returns *something* honest.
The ordinal CLMM (R) is a further layer, added later. Needs the ``stats`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cafe.judging.ratings import Ratings


@dataclass
class Effects:
    """Per-factor significance + variance attribution, plus pairwise effect sizes."""

    model: str = ""
    alpha: float = 0.05
    n_obs: int = 0
    terms: list[dict[str, Any]] = field(default_factory=list)       # factor, df, F, p, partial_eta_sq, significant
    pairwise_d: list[dict[str, Any]] = field(default_factory=list)  # factor, level_a/b, d, ci_low/high
    warnings: list[str] = field(default_factory=list)

    @property
    def significant_factors(self) -> list[str]:
        return [t["factor"] for t in self.terms if t.get("significant")]

    def show(self) -> str:
        lines = [f"model: {self.model}   (n={self.n_obs}, α={self.alpha})", ""]
        if self.terms:
            lines.append("factor effects (is it real? how much variance?):")
            lines.append(f"  {'factor':<16}{'F':>9}{'p':>10}{'partial η²':>13}   significant")
            for t in sorted(self.terms, key=lambda x: (x.get("partial_eta_sq") or 0), reverse=True):
                F = "-" if t["F"] is None else f"{t['F']:.2f}"
                p = "-" if t["p"] is None else f"{t['p']:.4f}"
                eta = "-" if t["partial_eta_sq"] is None else f"{t['partial_eta_sq']:.3f}"
                star = "✓ yes" if t.get("significant") else "  no"
                lines.append(f"  {t['factor']:<16}{F:>9}{p:>10}{eta:>13}   {star}")
        if self.pairwise_d:
            lines.append("")
            lines.append("pairwise effect sizes (Cohen's d):")
            for d in self.pairwise_d:
                val = "-" if d["d"] is None else f"{d['d']:+.2f}"
                ci = "" if d["ci_low"] is None else f"  [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]"
                lines.append(f"  {d['factor']}: {d['level_a']} vs {d['level_b']}  d={val}{ci}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def _cohens_d(arr_a, arr_b):
    import numpy as np

    n_a, n_b = arr_a.size, arr_b.size
    if n_a < 2 or n_b < 2:
        return None, None, None
    var_a, var_b = arr_a.var(ddof=1), arr_b.var(ddof=1)
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled <= 0:
        return None, None, None
    d = (float(arr_a.mean()) - float(arr_b.mean())) / float(np.sqrt(pooled))
    se = float(np.sqrt((n_a + n_b) / (n_a * n_b) + d**2 / (2 * (n_a + n_b))))
    return d, d - 1.96 * se, d + 1.96 * se


def _pairwise_d(df, factors):

    out = []
    for col in factors:
        levels = sorted(str(x) for x in df[col].dropna().unique())
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                la, lb = levels[i], levels[j]
                a = df.loc[df[col].astype(str) == la, "verdict"].to_numpy(dtype=float)
                b = df.loc[df[col].astype(str) == lb, "verdict"].to_numpy(dtype=float)
                d, lo, hi = _cohens_d(a, b)
                out.append({"factor": col, "level_a": la, "level_b": lb, "d": d, "ci_low": lo, "ci_high": hi})
    return out


def _oneway(df, factors, warnings, alpha):
    import numpy as np
    from scipy.stats import f_oneway

    terms = []
    for col in factors:
        groups = [
            df.loc[df[col].astype(str) == lvl, "verdict"].to_numpy(dtype=float)
            for lvl in sorted(str(x) for x in df[col].dropna().unique())
        ]
        groups = [g for g in groups if g.size >= 2]
        if len(groups) < 2:
            continue
        try:
            F, p = f_oneway(*groups)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"f_oneway failed for {col}: {exc}")
            continue
        allv = np.concatenate(groups)
        grand = float(allv.mean())
        ssb = sum(g.size * (float(g.mean()) - grand) ** 2 for g in groups)
        sst = float(((allv - grand) ** 2).sum())
        eta = (ssb / sst) if sst > 0 else None
        terms.append(
            {
                "factor": col,
                "df": float(len(groups) - 1),
                "F": float(F) if F == F else None,
                "p": float(p) if p == p else None,
                "partial_eta_sq": eta,
                "significant": (p < alpha) if p == p else False,
            }
        )
    return terms


def fit_effects(ratings: "Ratings", *, alpha: float = 0.05) -> Effects:
    """Fit the mixed-effects model and return per-factor significance + η²."""
    import numpy as np  # noqa: F401
    import pandas as pd

    res = Effects(alpha=alpha)
    df = pd.DataFrame(ratings.to_records())
    if df.empty or "verdict" not in df.columns:
        res.model = "skipped (no verdicts)"
        return res
    df = df.dropna(subset=["verdict"]).copy()
    df["verdict"] = df["verdict"].astype(float)
    res.n_obs = len(df)

    usable = [
        c for c in ratings.factors
        if c in df.columns and df[c].astype("string").dropna().nunique() >= 2
    ]
    if not usable:
        res.model = "skipped (no factor varies across ≥2 levels)"
        res.warnings.append("need a factor with at least two observed levels")
        return res

    res.pairwise_d = _pairwise_d(df, usable)

    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        res.model = "one-way ANOVA (statsmodels unavailable)"
        res.terms = _oneway(df, usable, res.warnings, alpha)
        return res

    formula = "verdict ~ " + " + ".join(f"C(Q('{c}'))" for c in usable)

    # Mixed model needs ≥2 questions to estimate the random intercept.
    if "input_id" in df.columns and df["input_id"].nunique() >= 2:
        try:
            smf.mixedlm(formula, df, groups=df["input_id"]).fit(method="lbfgs", reml=False, disp=False)
            res.model = "MixedLM (random intercept: question) + Type-II ANOVA"
        except Exception as exc:  # noqa: BLE001
            res.warnings.append(f"mixed model did not fit ({exc}); using one-way ANOVA")
            res.model = "one-way ANOVA (mixed model failed)"
            res.terms = _oneway(df, usable, res.warnings, alpha)
            return res
    else:
        res.warnings.append("only one question group; using one-way ANOVA")
        res.model = "one-way ANOVA (need ≥2 questions for a random effect)"
        res.terms = _oneway(df, usable, res.warnings, alpha)
        return res

    # Per-factor F / p / partial η² via Type-II ANOVA on the OLS counterpart.
    try:
        ols = smf.ols(formula, df).fit()
        table = sm.stats.anova_lm(ols, typ=2)
        ss_resid = float(table.loc["Residual", "sum_sq"])
        for c in usable:
            term = f"C(Q('{c}'))"
            if term not in table.index:
                continue
            ss = float(table.loc[term, "sum_sq"])
            F = float(table.loc[term, "F"])
            p = float(table.loc[term, "PR(>F)"])
            eta = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else None
            res.terms.append(
                {
                    "factor": c,
                    "df": float(table.loc[term, "df"]),
                    "F": F if F == F else None,
                    "p": p if p == p else None,
                    "partial_eta_sq": eta,
                    "significant": (p < alpha) if p == p else False,
                }
            )
    except Exception as exc:  # noqa: BLE001
        res.warnings.append(f"ANOVA failed ({exc})")
    return res
