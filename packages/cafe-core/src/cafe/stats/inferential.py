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


def _estimable_terms(df, factors: list[str], order: int):
    """The fixed-effects terms up to interaction ``order`` that the design can estimate, as
    ``(terms, dropped)``: tuples of factor names, main effects first, and the interactions
    left out. ``None`` if it can't be checked because statsmodels isn't installed.

    An interaction is left out when it adds no rank to the design given every other term
    still in it — it is aliased (confounded with other terms), e.g. in a fractional factorial
    below resolution V, or when the level combinations it would be estimated from have no
    answers (the empty cell of a 2 × 2 table); one that keeps some of its degrees of freedom
    stays. Orders are checked from the highest down, so an aliased three-way term goes first
    and the two-way terms are then judged without it. Main effects are always kept (the
    ANOVA reports one that adds no rank as not estimable). Shared by the Gaussian and
    ordinal layers, so their reports agree.
    """
    import warnings
    from itertools import combinations

    import numpy as np

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None

    top = order if order >= 2 and len(factors) >= 2 else 1
    terms = [c for k in range(1, top + 1) for c in combinations(factors, k)]
    if top < 2:
        return terms, []
    dfs, safe, _ = _safe_factor_frame(df, factors)
    main = " + ".join(f"C(Q('{safe[c]}'))" for c in factors)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.ols(f"verdict ~ ({main})**{top}", dfs)
    except Exception:  # noqa: BLE001 — can't build the design: let the model fit report it
        return terms, []
    x = np.asarray(model.exog)
    cols: dict[tuple, list[int]] = {}
    for j, name in enumerate(model.exog_names):  # () is the intercept
        cols.setdefault(tuple(c for c in factors if f"C(Q('{safe[c]}'))" in name), []).append(j)

    def rank(ts: list[tuple]) -> int:
        return int(np.linalg.matrix_rank(x[:, [j for t in [(), *ts] for j in cols.get(t, [])]]))

    if rank(terms) == x.shape[1]:
        return terms, []
    kept, dropped = list(terms), []
    for k in range(top, 1, -1):
        full = rank(kept)
        out = [t for t in kept if len(t) == k and not any(set(t) < set(u) for u in kept)
               and rank([u for u in kept if u != t]) == full]
        kept = [t for t in kept if t not in out]
        dropped += out
    return kept, dropped


def _aliased_note(dropped: list[tuple], order_used: int) -> str:
    """The note for interactions :func:`_estimable_terms` left out of the model."""
    why = ("e.g. a fractional factorial below resolution V, or level combinations no "
           "configuration answered")
    if order_used < 2:
        return (f"interactions are aliased in this design ({why}) — they can't be separated, "
                "so only main effects are fit; read the design's alias structure to interpret them")
    names = ", ".join(" × ".join(t) for t in dropped)
    one = len(dropped) == 1
    return (f"{names} {'is' if one else 'are'} aliased in this design (confounded with other "
            f"terms: {why}) — {'it' if one else 'they'} can't be separated, so "
            f"{'it is' if one else 'they are'} left out of the model")


def _type2_by_comparison(fit, tokens: list[str]):
    """The Type-II ANOVA table of an OLS ``fit`` with a rank-deficient design (or an exact
    fit), from nested least-squares comparisons — laid out like
    ``sm.stats.anova_lm(fit, typ=2)``, which it equals for a full-rank design. ``tokens`` are
    the model's factors as they appear in its column names (``"C(Q('cafe_f0'))"``, the block
    ``"C(input_id)"``); a term is the set of tokens its columns share.

    anova_lm counts every column of a term as a degree of freedom, so a term whose columns
    other terms already span (a factor whose levels were answered on different questions,
    which the question block spans; two factors that always change together) still gets
    df > 0 and an F-test of what those other terms explain. And it backs each SS out of its
    F-test, which divides by the residual variance: an exact fit (zero residual SS) gives NaN
    or rounding noise. Here, as R's ``car::Anova(type=2)`` does for a model with aliased
    coefficients, a term's SS = SSR(model without the term and the terms containing it) −
    SSR(model without the terms containing it), which needs no variance estimate, and its df
    is the rank the term adds. A term that adds none (df 0) is not estimable: its ``sum_sq``,
    ``F`` and ``PR(>F)`` are NaN. Raises ``ValueError`` when no residual df is left (like
    anova_lm).
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    X, y = np.asarray(fit.model.exog), np.asarray(fit.model.endog)
    cols: dict[frozenset, list[int]] = {}
    for j, name in enumerate(fit.model.exog_names):
        cols.setdefault(frozenset(t for t in tokens if t in name), []).append(j)

    def ssr_rank(terms: list[frozenset]) -> tuple[float, int]:
        x = X[:, [j for t in terms for j in cols[t]]]
        r = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        return float(r @ r), int(np.linalg.matrix_rank(x))

    ssr_full, rank_full = ssr_rank(list(cols))
    df_resid = len(y) - rank_full
    if df_resid <= 0:
        raise ValueError("no residual degrees of freedom")
    mse = ssr_full / df_resid
    rows = {}
    for t in cols:
        if not t:  # the intercept
            continue
        with_t = [u for u in cols if not t < u]  # drop the terms containing t
        ssr_with, rank_with = ssr_rank(with_t)
        ssr_without, rank_without = ssr_rank([u for u in with_t if u != t])
        df = rank_with - rank_without
        name = ":".join(tok for tok in tokens if tok in t)
        if df <= 0:
            rows[name] = (float("nan"), 0.0, float("nan"), float("nan"))
            continue
        ss = max(0.0, ssr_without - ssr_with)
        F = (ss / df) / mse if mse > 0 else float("nan")  # mse 0: the exact-fit path scores it
        rows[name] = (ss, float(df), F, float(stats.f.sf(F, df, df_resid)) if F == F else F)
    rows["Residual"] = (ssr_full, float(df_resid), float("nan"), float("nan"))
    return pd.DataFrame.from_dict(rows, orient="index", columns=["sum_sq", "df", "F", "PR(>F)"])


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
    most), ``3`` = up to three-way. It's auto-capped at the number of factors. An
    interaction the design can't estimate — aliased with other terms, adding nothing they
    don't already fit, e.g. in a fractional factorial below resolution V or with an empty
    cell in a 2 × 2 table of its levels — is left out of the model (checked from the highest
    order down) and ``warnings`` names it; if the interaction model can't be fit on the data
    at all, it falls back to main effects only.

    With two or more questions the F-tests are blocked by question, also when some
    configurations have no answer to some questions (a failed generation, an unparseable
    verdict): such a question just adds fewer within-question comparisons.

    Returns
    -------
    Effects
        Per-term ``F`` / ``p`` / ``partial_eta_sq`` / ``significant``, pairwise Cohen's d,
        variance components and notes. On an **exact fit** (zero residual variance: every
        answer within a configuration got the same score) the F-tests are degenerate, so a
        term with any effect is reported as ``F=inf``, ``p=0.0``, ``partial_eta_sq=1.0``
        (significant) and a term with no effect as ``None`` (not significant); ``model``
        and ``warnings`` say so. A term that adds no degrees of freedom of its own is **not
        estimable** — one confounded with the questions (its levels were answered on
        different questions, so the question block already explains whatever it might) or
        with other factors (two factors that always change together): it has ``df=0`` and
        ``F`` / ``p`` / ``partial_eta_sq`` ``None`` (not significant), and ``warnings`` names
        it. A term's ``df`` is the number of degrees of freedom it adds to the model (fewer
        than its levels imply when some level combinations have no answers).
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

    # Block by question whenever there are ≥2 questions. A question only some configurations
    # answered (e.g. one generation failed) just fits its own block parameter: the factors are
    # still compared within the questions their levels share.
    multi_q = "input_id" in df.columns and df["input_id"].nunique() >= 2
    block = "C(input_id)" if multi_q else ""
    tokens = [f"C(Q('{safe[c]}'))" for c in usable] + ([block] if block else [])

    if not multi_q:
        res.warnings.append("only one question group; using one-way ANOVA")
        res.model = "one-way ANOVA (need ≥2 questions to block by question)"
        res.formula = " · ".join(f"verdict ~ {c}" for c in usable)  # one test per factor
        res.terms = _oneway(df, usable, res.warnings, alpha)
        return res

    # The terms the design can estimate. An interaction aliased with other terms (a fractional
    # factorial below resolution V: at resolution IV model×sabotage IS verbosity×hedge; or an
    # empty cell in a 2 × 2 table of its levels) can't be separated — statsmodels would
    # silently split its variance into meaningless parts — so it is left out of the model.
    terms, dropped = _estimable_terms(df, usable, order)
    order_used = max(len(t) for t in terms)
    if dropped:
        res.warnings.append(_aliased_note(dropped, order_used))

    def _terms(o: int) -> list[tuple]:
        return [t for t in terms if len(t) <= o]

    def _formula(o: int, blocked: bool = True) -> str:
        rhs = " + ".join(":".join(f"C(Q('{safe[c]}'))" for c in t) for t in _terms(o))
        return f"verdict ~ {block} + {rhs}" if blocked and block else f"verdict ~ {rhs}"

    def _display_formula(o: int, blocked: bool = True) -> str:
        if o < 2 or len(usable) < 2:
            fixed = " + ".join(usable)
        elif any(len(t) <= o for t in dropped):  # some interactions left out: list the rest
            fixed = " + ".join(":".join(t) for t in _terms(o))
        else:
            fixed = f"({' + '.join(usable)})^{o}"
        return f"verdict ~ question + {fixed}" if blocked and block else f"verdict ~ {fixed}"

    res.formula = _display_formula(order_used)

    # How much the questions differ: the random-intercept model's variance components. Its
    # fixed part is the main effects; where some of their columns are confounded with others,
    # only the columns that add rank are kept (as lme4 does). That is the same model, which
    # statsmodels can't fit with the redundant columns in ("Singular matrix").
    vc_formula, vc_frame = _formula(1, blocked=False), dfs
    x = np.asarray(smf.ols(vc_formula, dfs).exog)
    if np.linalg.matrix_rank(x) < x.shape[1]:
        keep: list[int] = []
        for j in range(x.shape[1]):  # column 0, the intercept, is always kept
            if np.linalg.matrix_rank(x[:, [*keep, j]]) > len(keep):
                keep.append(j)
        vc_frame = dfs[["verdict", "input_id"]].assign(**{f"x{j}": x[:, j] for j in keep[1:]})
        vc_formula = "verdict ~ " + (" + ".join(f"x{j}" for j in keep[1:]) or "1")
    mfit = _fit_random_intercept(smf, vc_formula, vc_frame)
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

    def _ols(o: int, blocked: bool):
        """OLS fit at interaction order ``o`` and whether its design is rank-deficient."""
        f = smf.ols(_formula(o, blocked), dfs).fit()
        x = f.model.exog
        return f, bool(np.linalg.matrix_rank(x) < x.shape[1])

    def _anova(o: int, blocked: bool):
        """The fit at order ``o`` and its Type-II table. A rank-deficient design's table comes
        from nested comparisons, which give a term that adds no rank of its own — confounded
        with the questions or with other factors — 0 df: anova_lm would test what the terms
        it is confounded with explain as its effect."""
        f, deficient = _ols(o, blocked)
        return f, (_type2_by_comparison(f, tokens) if deficient else sm.stats.anova_lm(f, typ=2))

    def _not_estimable(tab) -> list[tuple]:
        """The factor terms of a Type-II table that add no degrees of freedom (df 0)."""
        return [tuple(c for c in usable if f"Q('{safe[c]}')" in n) for n in tab.index
                if "Q('" in n and float(tab.loc[n, "df"]) == 0]

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        # Per-term F / p / partial η² via Type-II ANOVA, blocked by question, with the
        # estimable terms up to the requested interaction order.
        table, exact, use_block = None, False, bool(block)
        try:
            fit, table = _anova(order_used, use_block)
        except Exception:  # noqa: BLE001
            if order_used >= 2:
                order_used = 1
                res.warnings.append("interactions not estimable on this data — main effects only")
                try:
                    # Rebind `fit`: the residuals/fitted attached below must come from the
                    # main-effects model the table reports, not the abandoned interaction fit.
                    fit, table = _anova(1, use_block)
                except Exception as exc:  # noqa: BLE001
                    res.warnings.append(f"ANOVA failed ({exc})")
            else:
                res.warnings.append("ANOVA failed")

        if table is None:
            res.model = "ANOVA failed"
            res.terms = _oneway(df, usable, res.warnings, alpha)
        else:
            # Terms confounded with the questions: no df with the block, some without it.
            q_confounded: list[tuple] = []
            zero = _not_estimable(table)
            if zero and use_block:
                fit_u, deficient_u = _ols(order_used, False)
                zero_u = _not_estimable(_type2_by_comparison(fit_u, tokens)) if deficient_u else []
                q_confounded = [t for t in zero if t not in zero_u]
            ss_resid = float(table.loc["Residual", "sum_sq"])
            ss_total = float(((dfs["verdict"] - dfs["verdict"].mean()) ** 2).sum())
            # Exact fit: the model explains every score, so the residual variance is zero and the
            # table's F-tests are 0/0 (NaN) or rounding noise (needs residual df to mean anything).
            exact = (float(table.loc["Residual", "df"]) > 0
                     and ss_resid <= _EXACT_FIT_RTOL * ss_total)
            if exact and use_block and not q_confounded:
                # With few answers per question the block alone can make the fit exact by chance
                # (a null factor would read F=inf); unless the factors alone fit exactly too,
                # test without the block instead. (Not when a term is confounded with the
                # questions: without the block it would be tested on question difficulty.)
                fit_u = smf.ols(_formula(order_used, False), dfs).fit()
                if float(fit_u.ssr) > _EXACT_FIT_RTOL * ss_total:
                    use_block, exact = False, False
                    fit, table = _anova(order_used, False)
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
                # Score each term from its exact Type-II SS (nested comparisons need no variance
                # estimate) instead of the degenerate table.
                table = _type2_by_comparison(fit, tokens)
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
                    if ss > _EXACT_FIT_RTOL * ss_total:
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
            zero = _not_estimable(table)
            by_questions = [" × ".join(t) for t in zero if use_block and t in q_confounded]
            by_factors = [" × ".join(t) for t in zero if not (use_block and t in q_confounded)]
            if by_questions:
                res.warnings.append(
                    f"{', '.join(by_questions)}: not estimable — confounded with the questions "
                    "(its levels were answered on different questions), so its effect can't be "
                    "told apart from question difficulty and it gets no F-test; answer the same "
                    "questions with each level to test it"
                )
            if by_factors:
                res.warnings.append(
                    f"{', '.join(by_factors)}: not estimable — confounded with other terms in "
                    "this design (one factor determines another), so each adds no degrees of "
                    "freedom of its own and gets no F-test; read the confounded factors together"
                )

    return res
