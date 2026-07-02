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
    formula: str = ""
    alpha: float = 0.05
    n_obs: int = 0
    terms: list[dict[str, Any]] = field(default_factory=list)       # factor, df, F, p, partial_eta_sq, significant
    pairwise_d: list[dict[str, Any]] = field(default_factory=list)  # factor, level_a/b, d, ci_low/high
    warnings: list[str] = field(default_factory=list)

    @property
    def significant_factors(self) -> list[str]:
        return [t["factor"] for t in self.terms if t.get("significant")]

    def show(self) -> str:
        from cafe.stats._format import SIG_LEGEND, sig_code

        lines = [self.formula or self.model, f"  {self.model}   (n={self.n_obs}, α={self.alpha})", ""]
        if self.terms:
            rows = sorted(self.terms, key=lambda x: (x.get("partial_eta_sq") or 0), reverse=True)
            w = max([len("term")] + [len(t["factor"]) for t in rows])
            lines.append("per-term effects  (F-test, p, partial η²;  '×' = interaction):")
            lines.append(f"  {'term':<{w}}{'F':>9}{'p':>11}{'partial η²':>13}     ")
            for t in rows:
                if t["F"] is None:
                    F = "-"
                elif t["F"] >= 1e4:  # degenerate/separated fit — avoid a 30-digit number
                    F = f"{t['F']:.0e}"
                else:
                    F = f"{t['F']:.2f}"
                p = "-" if t["p"] is None else f"{t['p']:.4f}"
                eta = "-" if t["partial_eta_sq"] is None else f"{t['partial_eta_sq']:.3f}"
                lines.append(f"  {t['factor']:<{w}}{F:>9}{p:>11}{eta:>13}   {sig_code(t['p'])}")
            lines.append(f"  {SIG_LEGEND}")
        if self.pairwise_d:
            lines.append("")
            lines.append("effect sizes — Cohen's d (magnitude of the gap; 0.2 small, 0.5 medium, 0.8 large):")
            for d in self.pairwise_d:
                val = " n/a" if d["d"] is None else f"{d['d']:+.2f}"
                ci = "" if d["ci_low"] is None else f"   95% CI [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]"
                lines.append(f"  {d['factor']}: {d['level_a']} vs {d['level_b']}   d = {val}{ci}")
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"note: {w}")
        return "\n".join(lines)


def _safe_factor_frame(df, factors: list[str]):
    """Rename factor columns to collision-proof patsy identifiers and return
    ``(renamed_df, real->safe, safe->real)``. A factor named like a patsy builtin (``C``,
    ``Q``, ``I``) would otherwise shadow that function in the formula namespace and crash
    (``'Series' object is not callable``). Screening designs often use letter names, so this
    matters here."""
    safe = {c: f"cafe_f{i}" for i, c in enumerate(factors)}
    return df.rename(columns=safe), safe, {v: k for k, v in safe.items()}


def _design_rank_deficient(df, factors: list[str], order: int) -> bool:
    """Whether the fixed-effects design at this interaction ``order`` is rank-deficient —
    true for an **aliased** design (e.g. a fractional factorial below resolution V), where
    interaction columns are collinear and cannot be separated. Used by both the Gaussian
    and ordinal layers to avoid reporting un-estimable interactions."""
    if order < 2 or len(factors) < 2:
        return False
    import numpy as np
    import statsmodels.formula.api as smf

    dfs, safe, _ = _safe_factor_frame(df, factors)
    main = " + ".join(f"C(Q('{safe[c]}'))" for c in factors)
    try:
        exog = smf.ols(f"verdict ~ ({main})**{order}", dfs).fit().model.exog
    except Exception:  # noqa: BLE001
        return False
    return bool(np.linalg.matrix_rank(exog) < exog.shape[1])


def _degenerate(w) -> bool:
    """Whether a captured warning indicates a near-singular / non-converged fit."""
    msg = str(getattr(w, "message", w)).lower()
    return any(k in msg for k in ("singular", "converge", "positive definite", "boundary"))


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


def fit_effects(ratings: "Ratings", *, alpha: float = 0.05, interactions: int = 2) -> Effects:
    """Fit the mixed-effects model and return per-term significance + η².

    ``interactions`` is the maximum interaction order to include: ``1`` = main effects
    only, ``2`` = also two-way (e.g. model×prompt — the default, since two-way captures
    most), ``3`` = up to three-way. It's auto-capped at the number of factors, and falls
    back to main-effects-only if the interaction model can't be estimated on the data.
    """
    import numpy as np  # noqa: F401

    from cafe.stats._frame import analysis_frame

    res = Effects(alpha=alpha)
    df = analysis_frame(ratings)  # one row per answer (judge replications averaged)
    if df.empty or "verdict" not in df.columns:
        res.model = "skipped (no verdicts)"
        return res
    df["verdict"] = df["verdict"].astype(float)
    res.n_obs = len(df)

    if df["verdict"].nunique() < 2:
        # Every answer scored identically — there is no variance for any factor to
        # explain (a degenerate fit would otherwise report a spurious effect).
        res.model = "skipped (no variance in verdict — every answer scored the same)"
        res.warnings.append("all verdicts identical; no effect to estimate")
        return res

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

    order = max(1, min(interactions, len(usable)))
    # Fit on collision-proof column names so a factor named like a patsy builtin (C/Q/I)
    # can't shadow the formula function; the display formula + term labels keep real names.
    dfs, safe, _back = _safe_factor_frame(df, usable)
    main_terms = " + ".join(f"C(Q('{safe[c]}'))" for c in usable)

    def _formula(o: int) -> str:
        return f"verdict ~ ({main_terms})**{o}" if o >= 2 and len(usable) >= 2 else f"verdict ~ {main_terms}"

    def _display_formula(o: int) -> str:
        fixed = f"({' + '.join(usable)})^{o}" if o >= 2 and len(usable) >= 2 else " + ".join(usable)
        return f"verdict ~ (1 | input_id) + {fixed}"  # random effect first, human-readable

    res.formula = _display_formula(order)

    # Mixed model needs ≥2 questions to estimate the random intercept.
    if not ("input_id" in df.columns and df["input_id"].nunique() >= 2):
        res.warnings.append("only one question group; using one-way ANOVA")
        res.model = "one-way ANOVA (need ≥2 questions for a random effect)"
        res.terms = _oneway(df, usable, res.warnings, alpha)
        return res

    # statsmodels emits a flurry of singular/convergence warnings on small or
    # near-degenerate data. Capture them quietly and translate to ONE clear note,
    # rather than leaking the raw stack-trace-like spam to the user.
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        # The main-effects mixedlm just confirms the random intercept is identifiable.
        # If it isn't, we keep the interaction formula but drop to fixed-effects ANOVA
        # (rather than one-way, which would silently lose the interaction terms).
        mixed_ok = True
        try:
            smf.mixedlm(_formula(1), dfs, groups=dfs["input_id"]).fit(method="lbfgs", reml=False, disp=False)
        except Exception:  # noqa: BLE001
            mixed_ok = False
            res.warnings.append("random intercept not estimable; using fixed-effects ANOVA")

        # Per-term F / p / partial η² via Type-II ANOVA on the OLS counterpart, at the
        # requested interaction order — falling back to main effects if it won't estimate.
        order_used, table = order, None
        try:
            fit = smf.ols(_formula(order), dfs).fit()
            # A fractional factorial (or any aliased design) makes interaction columns
            # collinear — e.g. at resolution IV model×sabotage IS verbosity×hedge. statsmodels
            # would silently split their variance into meaningless separate terms, so detect
            # the rank deficiency and drop to main effects rather than report bogus interactions.
            if order >= 2:
                exog = fit.model.exog
                if np.linalg.matrix_rank(exog) < exog.shape[1]:
                    order_used = 1
                    res.formula = _display_formula(1)
                    res.warnings.append(
                        "interactions are aliased in this design (e.g. a fractional factorial "
                        "below resolution V) — they can't be separated, so only main effects are "
                        "fit; read the design's alias structure to interpret them"
                    )
                    fit = smf.ols(_formula(1), dfs).fit()
            table = sm.stats.anova_lm(fit, typ=2)
        except Exception:  # noqa: BLE001
            if order_used >= 2:
                order_used = 1
                res.formula = _display_formula(1)
                res.warnings.append("interactions not estimable on this data — main effects only")
                try:
                    table = sm.stats.anova_lm(smf.ols(_formula(1), dfs).fit(), typ=2)
                except Exception as exc:  # noqa: BLE001
                    res.warnings.append(f"ANOVA failed ({exc})")
            else:
                res.warnings.append("ANOVA failed")

        if table is None:
            res.model = "ANOVA failed"
            res.terms = _oneway(df, usable, res.warnings, alpha)
        else:
            base = "MixedLM (random intercept: question)" if mixed_ok else "fixed-effects model"
            res.model = (f"{base} + Type-II ANOVA"
                         + (f", up to {order_used}-way" if order_used >= 2 else ""))
            ss_resid = float(table.loc["Residual", "sum_sq"])
            for name in table.index:
                if name in ("Residual", "Intercept"):
                    continue
                cols = [c for c in usable if f"Q('{safe[c]}')" in name]
                if not cols:
                    continue
                ss = float(table.loc[name, "sum_sq"])
                F = float(table.loc[name, "F"])
                p = float(table.loc[name, "PR(>F)"])
                eta = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else None
                res.terms.append(
                    {
                        "factor": " × ".join(cols),
                        "interaction": len(cols) > 1,
                        "df": float(table.loc[name, "df"]),
                        "F": F if F == F else None,
                        "p": p if p == p else None,
                        "partial_eta_sq": eta,
                        "significant": (p < alpha) if p == p else False,
                    }
                )

    if any(_degenerate(w) for w in caught):
        res.warnings.append(
            "model was near-singular (little between-question variance) — treat p-values as "
            "unstable; the effect sizes (Cohen's d) are the more reliable signal here"
        )
    return res
