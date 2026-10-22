"""Inferential statistics over judge ratings — the paper's spine.

Turns verdicts into *which factor matters, by how much, and is it real*:

- per-factor **F / p** and **partial η²** (the share of variance a factor explains) from a
  Type-II ANOVA **blocked by question** — every configuration answers the same questions,
  so question difficulty is taken out of the error term and configurations are compared
  within each question (in a balanced design these are the random-intercept model's tests),
- the **variance components** of the random-intercept model ``verdict ~ factors + (1 | question)``
  (how much questions differ, vs. the residual noise),
- pairwise **Cohen's d** effect sizes with CIs.

Robust by design: falls back to one-way ANOVA when there's only one question (or no
statsmodels) so it always returns *something* honest. The scale-matched R models (ordinal
CLMM, logistic GLMM) live in ``ordinal.py`` / ``logistic.py``. Needs the ``stats`` extra.
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
    #: Random-intercept / residual variance (from the mixed model), or ``None`` when only a
    #: fixed-effects fit was possible. ``{"random_intercept": σ², "residual": σ²}``.
    variance_components: dict[str, float] | None = None
    #: Share of all verdict variance the factors (and their modeled interactions) explain —
    #: from the model *without* the question block, so question difficulty doesn't count.
    r_squared: float | None = None
    #: Per-answer residuals / fitted values from the fit, aligned to the analysis frame.
    residuals: list[float] = field(default_factory=list, repr=False)
    fitted: list[float] = field(default_factory=list, repr=False)

    @property
    def significant_factors(self) -> list[str]:
        return [t["factor"] for t in self.terms if t.get("significant")]

    def to_df(self):
        """The per-term table as a tidy DataFrame (term, df, F, p, partial η², significant) —
        the data behind :meth:`show`, ready for a paper table. Needs the ``stats`` extra."""
        import pandas as pd

        cols = ["factor", "interaction", "df", "F", "p", "partial_eta_sq", "significant"]
        return pd.DataFrame([{c: t.get(c) for c in cols} for t in self.terms])

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
        if self.variance_components:
            vc = self.variance_components
            lines.append("")
            lines.append(
                f"variance components:  between-question σ² = {vc['random_intercept']:.3f}"
                f" · residual σ² = {vc['residual']:.3f}"
                f" · ICC = {vc['random_intercept'] / (vc['random_intercept'] + vc['residual']):.3f}"
            )
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


def _design_rank_deficient(df, factors: list[str], order: int) -> bool | None:
    """Whether the fixed-effects design at this interaction ``order`` is rank-deficient —
    true for an **aliased** design (e.g. a fractional factorial below resolution V), where
    interaction columns are collinear and cannot be separated. Used by both the Gaussian
    and ordinal layers to avoid reporting un-estimable interactions. ``None`` if it can't
    be checked because statsmodels isn't installed (``fit_clmm`` must still work then)."""
    if order < 2 or len(factors) < 2:
        return False
    import numpy as np

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None

    dfs, safe, _ = _safe_factor_frame(df, factors)
    main = " + ".join(f"C(Q('{safe[c]}'))" for c in factors)
    try:
        exog = smf.ols(f"verdict ~ ({main})**{order}", dfs).fit().model.exog
    except Exception:  # noqa: BLE001
        return False
    return bool(np.linalg.matrix_rank(exog) < exog.shape[1])


#: Relative to the total sum of squares, a residual (or term) sum of squares at or below this
#: counts as zero: floating-point rounding leaves ~1e-30 where the exact value is 0.
_EXACT_FIT_RTOL = 1e-10


def _exact_fit_ss(dfs, factors: list[str], safe: dict[str, str], order: int,
                  block: str = "") -> dict:
    """Type-II sum of squares per model term (keyed by its set of factors), from nested
    least-squares refits — for a model that fits the verdicts exactly (zero residual SS).

    statsmodels backs each term's SS out of its F-test, which divides by the residual
    variance: an exact zero gives an all-NaN table, and rounding noise gives F-ratios of
    noise over noise (a term with no effect at all can come out "significant"). Here
    SS(term) = SSR(model without the term and the terms containing it) − SSR(model without
    the terms containing it), which needs no variance estimate. ``block`` (e.g. the question
    block ``"C(input_id)"``) is kept in every model, as in the ANOVA it stands in for.
    """
    from itertools import combinations

    import statsmodels.formula.api as smf

    top = order if order >= 2 and len(factors) >= 2 else 1
    terms = [frozenset(c) for k in range(1, top + 1) for c in combinations(factors, k)]

    def ssr(model_terms: list[frozenset]) -> float:
        parts = ([block] if block else []) + [
            ":".join(f"C(Q('{safe[c]}'))" for c in factors if c in t) for t in model_terms
        ]
        return float(smf.ols(f"verdict ~ {' + '.join(parts) or '1'}", dfs).fit().ssr)

    out = {}
    for term in terms:
        kept = [t for t in terms if not term < t]  # drop the higher-order terms containing it
        out[term] = ssr([t for t in kept if t != term]) - ssr(kept)
    return out


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

    try:
        from scipy.stats import f_oneway
    except ImportError:  # pandas without the stats extra (e.g. cafe-core[llm,notebooks])
        f_oneway = None
        warnings.append(
            "scipy is not installed, so the one-way ANOVA has no F-tests / p-values (partial "
            "η² and Cohen's d are still shown); install the stats extra: "
            "pip install 'cafe-core[stats]'"
        )

    terms = []
    for col in factors:
        groups = [
            df.loc[df[col].astype(str) == lvl, "verdict"].to_numpy(dtype=float)
            for lvl in sorted(str(x) for x in df[col].dropna().unique())
        ]
        groups = [g for g in groups if g.size >= 2]
        if len(groups) < 2:
            continue
        if f_oneway is None:
            F = p = float("nan")  # reported as None (not significant)
        else:
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


#: Optimizers tried in turn for the random-intercept model. statsmodels' lbfgs can crash
#: ("Singular matrix") on data the others fit fine — on the paper's HotpotQA study it did,
#: while bfgs / powell / nm all matched R's lme4 (between-question σ² = 0.207).
_MIXEDLM_METHODS = ("lbfgs", "bfgs", "powell", "nm")


def _fit_random_intercept(smf, formula: str, dfs):
    """Fit ``formula`` with a random intercept per question (ML), trying each optimizer in
    :data:`_MIXEDLM_METHODS` until one converges. Returns the fit, or ``None`` if none did."""
    import warnings

    import numpy as np

    for method in _MIXEDLM_METHODS:
        # statsmodels' convergence chatter is judged below instead; llf is computed lazily,
        # so it is read inside the silenced block too.
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                fit = smf.mixedlm(formula, dfs, groups=dfs["input_id"]).fit(
                    method=method, reml=False, disp=False)
                ok = bool(getattr(fit, "converged", True)) and bool(np.isfinite(fit.llf))
            except Exception:  # noqa: BLE001
                continue
        if ok:
            return fit
    return None


def fit_effects(ratings: "Ratings", *, alpha: float = 0.05, interactions: int = 2) -> Effects:
    """Fit the mixed-effects model and return per-term significance + η².

    Needs statsmodels (the ``stats`` extra). Without it this falls back to a one-way ANOVA
    per factor, whose F-tests also need scipy: with neither installed the terms carry
    partial η² but no ``F`` / ``p`` (not significant), and ``warnings`` says so — nothing
    is raised.

    ``interactions`` is the maximum interaction order to include: ``1`` = main effects
    only, ``2`` = also two-way (e.g. model×prompt — the default, since two-way captures
    most), ``3`` = up to three-way. It's auto-capped at the number of factors, and falls
    back to main-effects-only if the interaction model can't be estimated on the data.

    Returns
    -------
    Effects
        Per-term ``F`` / ``p`` / ``partial_eta_sq`` / ``significant``, pairwise Cohen's d,
        variance components and notes. On an **exact fit** (zero residual variance: every
        answer within a configuration got the same score) the F-tests are degenerate, so a
        term with any effect is reported as ``F=inf``, ``p=0.0``, ``partial_eta_sq=1.0``
        (significant) and a term with no effect as ``None`` (not significant); ``model``
        and ``warnings`` say so.
    """
    import numpy as np

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

    # Blocking needs ≥2 questions, and questions crossed with configurations: if a question
    # was answered by only one configuration, its block would soak up that config's effect.
    multi_q = "input_id" in df.columns and df["input_id"].nunique() >= 2
    crossed = False
    if multi_q:
        config_key = dfs[[safe[c] for c in usable]].astype(str).agg("|".join, axis=1)
        crossed = bool((config_key.groupby(dfs["input_id"]).nunique() >= 2).all())
    block = "C(input_id)" if crossed else ""

    def _formula(o: int, blocked: bool = True) -> str:
        rhs = f"({main_terms})**{o}" if o >= 2 and len(usable) >= 2 else main_terms
        return f"verdict ~ {block} + {rhs}" if blocked and block else f"verdict ~ {rhs}"

    def _display_formula(o: int, blocked: bool = True) -> str:
        fixed = f"({' + '.join(usable)})^{o}" if o >= 2 and len(usable) >= 2 else " + ".join(usable)
        return f"verdict ~ question + {fixed}" if blocked and block else f"verdict ~ {fixed}"

    res.formula = _display_formula(order)

    if not multi_q:
        res.warnings.append("only one question group; using one-way ANOVA")
        res.model = "one-way ANOVA (need ≥2 questions to block by question)"
        res.formula = " · ".join(f"verdict ~ {c}" for c in usable)  # one test per factor
        res.terms = _oneway(df, usable, res.warnings, alpha)
        return res
    if not crossed:
        res.warnings.append(
            "some question was answered by only one configuration, so the F-tests can't be "
            "blocked by question: question difficulty stays in the error term, which makes "
            "them conservative"
        )

    # How much the questions differ: the random-intercept model's variance components.
    mfit = _fit_random_intercept(smf, _formula(1, blocked=False), dfs)
    if mfit is not None:
        re_var = float(np.asarray(mfit.cov_re).ravel()[0])
        scale = float(mfit.scale)
        res.variance_components = {"random_intercept": re_var, "residual": scale}
        if re_var <= 1e-6 * (re_var + scale):
            res.warnings.append(
                "between-question variance estimated at ≈ 0 (a boundary fit): the questions "
                "don't differ beyond what the factors explain"
            )
    else:
        res.warnings.append(
            "the between-question variance couldn't be estimated (the random-intercept model "
            "didn't converge with any optimizer); the F-tests don't depend on it"
        )

    # statsmodels emits a flurry of warnings on small or near-degenerate data; the checks
    # below (rank, exact fit) turn the ones that matter into clear notes.
    import warnings as _warnings

    confounded_note = (
        "a factor is confounded with the questions (configurations answered different "
        "questions), so the F-tests can't be blocked by question: question difficulty stays "
        "in the error term, which makes them conservative"
    )

    def _ols(o: int, blocked: bool):
        """OLS fit at interaction order ``o`` and whether its design is rank-deficient."""
        f = smf.ols(_formula(o, blocked), dfs).fit()
        x = f.model.exog
        return f, bool(np.linalg.matrix_rank(x) < x.shape[1])

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        # Per-term F / p / partial η² via Type-II ANOVA, blocked by question, at the requested
        # interaction order. A rank-deficient design can't separate its terms (statsmodels would
        # silently split their variance into meaningless parts), so: if the question block is
        # what breaks it, drop the block; if interactions are aliased (a fractional factorial
        # below resolution V: at resolution IV model×sabotage IS verbosity×hedge), fit main
        # effects only.
        order_used, table, exact, use_block = order, None, False, bool(block)
        try:
            fit, deficient = _ols(order, use_block)
            if deficient and use_block and not _ols(order, False)[1]:
                use_block, (fit, deficient) = False, _ols(order, False)
                res.warnings.append(confounded_note)
            if deficient and order >= 2:
                order_used = 1
                res.warnings.append(
                    "interactions are aliased in this design (e.g. a fractional factorial "
                    "below resolution V) — they can't be separated, so only main effects are "
                    "fit; read the design's alias structure to interpret them"
                )
                fit, deficient = _ols(1, use_block)
                if deficient and use_block and not _ols(1, False)[1]:
                    use_block, (fit, deficient) = False, _ols(1, False)
                    res.warnings.append(confounded_note)
            if deficient:
                res.warnings.append(
                    "some factors are confounded with each other in this design (one determines "
                    "another), so their separate effects can't be estimated — read them together"
                )
            table = sm.stats.anova_lm(fit, typ=2)
        except Exception:  # noqa: BLE001
            if order_used >= 2:
                order_used = 1
                res.warnings.append("interactions not estimable on this data — main effects only")
                try:
                    # Rebind `fit`: the residuals/fitted attached below must come from the
                    # main-effects model the table reports, not the abandoned interaction fit.
                    fit = smf.ols(_formula(1, use_block), dfs).fit()
                    table = sm.stats.anova_lm(fit, typ=2)
                except Exception as exc:  # noqa: BLE001
                    res.warnings.append(f"ANOVA failed ({exc})")
            else:
                res.warnings.append("ANOVA failed")

        if table is None:
            res.model = "ANOVA failed"
            res.terms = _oneway(df, usable, res.warnings, alpha)
        else:
            ss_resid = float(table.loc["Residual", "sum_sq"])
            ss_total = float(((dfs["verdict"] - dfs["verdict"].mean()) ** 2).sum())
            # Exact fit: the model explains every score, so the residual variance is zero and the
            # table's F-tests are 0/0 (NaN) or rounding noise (needs residual df to mean anything).
            exact = (float(table.loc["Residual", "df"]) > 0
                     and ss_resid <= _EXACT_FIT_RTOL * ss_total)
            if exact and use_block:
                # With few answers per question the block alone can make the fit exact by chance
                # (a null factor would read F=inf); unless the factors alone fit exactly too,
                # test without the block instead.
                fit_u = smf.ols(_formula(order_used, False), dfs).fit()
                if float(fit_u.ssr) > _EXACT_FIT_RTOL * ss_total:
                    use_block, fit, exact = False, fit_u, False
                    table = sm.stats.anova_lm(fit, typ=2)
                    ss_resid = float(table.loc["Residual", "sum_sq"])
                    res.warnings.append(
                        "with the question as a block the model explains every score exactly "
                        "(too few answers per question to test that way), so the F-tests don't "
                        "block by question"
                    )
            res.formula = _display_formula(order_used, use_block)
            # Per-answer residuals / fitted (aligned to the analysis frame rows).
            try:
                res.residuals = [float(x) for x in np.asarray(fit.resid)]
                res.fitted = [float(x) for x in np.asarray(fit.fittedvalues)]
            except Exception:  # noqa: BLE001
                pass
            res.model = ("Type-II ANOVA, blocked by question" if use_block else "Type-II ANOVA")
            res.model += f", up to {order_used}-way" if order_used >= 2 else ""
            # R² of the factors alone (no question block): the share of verdict variance the
            # configuration explains.
            try:
                if ss_total > 0:
                    ssr_f = float(smf.ols(_formula(order_used, blocked=False), dfs).fit().ssr)
                    res.r_squared = max(0.0, 1.0 - ssr_f / ss_total)
            except Exception:  # noqa: BLE001
                pass
            if exact:
                # Score each term from its exact Type-II SS instead of the degenerate table.
                exact_ss = _exact_fit_ss(dfs, usable, safe, order_used,
                                         block=block if use_block else "")
                res.model += " — exact fit (zero residual variance)"
                res.warnings.append(
                    "zero residual variance: the factors"
                    + (" and the questions" if use_block else "")
                    + " explain every score exactly, so the F-tests are degenerate — a term "
                    "with any effect is reported as F=inf, p=0, partial η²=1 and a term with "
                    "none as undefined (not significant); compare the configuration means directly"
                )
            for name in table.index:
                if name in ("Residual", "Intercept"):
                    continue
                cols = [c for c in usable if f"Q('{safe[c]}')" in name]
                if not cols:  # the question block (or anything else that isn't a factor)
                    continue
                ss = float(table.loc[name, "sum_sq"])
                F = float(table.loc[name, "F"])
                p = float(table.loc[name, "PR(>F)"])
                eta = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else None
                if exact:
                    # SS/0: an effect is infinitely large against zero noise; no effect is 0/0.
                    if exact_ss.get(frozenset(cols), 0.0) > _EXACT_FIT_RTOL * ss_total:
                        F, p, eta = float("inf"), 0.0, 1.0
                    else:
                        F, p, eta = float("nan"), float("nan"), None
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

    return res
