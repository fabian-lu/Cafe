"""Logistic model for **binary** judge verdicts — the statistically correct model for a
pass/fail (0/1) rubric.

A binary outcome must be modelled on the log-odds scale, not as a Gaussian mean (a
linear-probability model can predict impossible probabilities and mis-states
significance). Because the same questions recur across configurations, the effect of a
factor should be estimated *net of* question difficulty — a per-question random intercept.
So the primary model is a **logistic mixed model** (GLMM: ``verdict ~ factors +
(1|question)``, binomial/logit) fit in R via ``lme4::glmer`` — the binary analogue of the
ordinal CLMM and the numeric linear mixed model, all three conditional random-intercept
models.

R is a *runtime* prerequisite for the GLMM (same as the CLMM). When R or ``lme4`` is
missing, ``fit_logistic`` falls back to a self-contained **statsmodels** logistic (GEE,
cluster-robust by question, or plain logistic) and says so. Either way a binary rubric
always gets a logistic model. Needs the ``stats`` extra (statsmodels) for the fallback.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cafe.judging.ratings import Ratings

# One ``C(Q('<column>'))[T.<level>]`` group of a statsmodels term; interaction terms join
# groups with ':'. A level may itself contain ':' or ']' (ollama tags like ``llama3:8b``), so
# a group ends only at a ']' that is followed by the next group or by the end of the name.
_TERM_RE = re.compile(r"C\(Q\('([^']+)'\)\)\[T\.(.*?)\](?=:C\(Q\('|$)")


def _parse_statsmodels_term(name: str, back: dict[str, str]) -> tuple[str, str | None, bool]:
    """Turn a statsmodels term like ``C(Q('cafe_f0'))[T.good]`` into
    (``"model=good"``, ``"model"``, is_interaction), mapping the safe column name back to the
    real factor via ``back`` (safe->real). Interactions (``a:b``) join with ×. The groups are
    matched whole rather than split on ``:``, so ``[T.llama3:8b]`` stays one level."""
    groups = list(_TERM_RE.finditer(name))
    if not groups:
        return name, None, False
    labels = [f"{back.get(g.group(1), g.group(1))}={g.group(2)}" for g in groups]
    first_factor = back.get(groups[0].group(1), groups[0].group(1))
    return " × ".join(labels), first_factor, len(groups) > 1


@dataclass
class Logistic:
    """Result of the binary logistic model (or why it couldn't be fit)."""

    available: bool = False
    reason: str | None = None
    alpha: float = 0.05
    model: str = ""
    formula: str = ""
    n_obs: int = 0
    converged: bool | None = None    # GLMM: lme4's convergence verdict; False -> p is None
    terms: list[dict[str, Any]] = field(default_factory=list)  # label, factor, interaction, coef, odds_ratio, p, significant
    warnings: list[str] = field(default_factory=list)
    #: The GLMM's per-question random intercept on the latent logit scale (like R's
    #: ``VarCorr(fit)``): ``{"random_intercept": σ_u², "random_intercept_sd": σ_u,
    #: "residual": π²/3, "icc": σ_u² / (σ_u² + π²/3)}``. ``None`` for the statsmodels
    #: backends, which fit no random intercept.
    variance_components: dict[str, float] | None = None

    @property
    def significant_factors(self) -> list[str]:
        return sorted({t["factor"] for t in self.terms if t.get("significant") and t.get("factor")})

    def to_df(self):
        """The fixed-effects table as a tidy DataFrame (term, log-odds, odds_ratio, p,
        significant) — the data behind :meth:`show`. Needs the ``stats`` extra."""
        import pandas as pd

        cols = ["label", "factor", "interaction", "coef", "odds_ratio", "p", "significant"]
        return pd.DataFrame([{c: t.get(c) for c in cols} for t in self.terms])

    def show(self) -> str:
        from cafe.stats._format import SIG_LEGEND, sig_code

        if not self.available:
            return f"logistic model unavailable: {self.reason}"
        labels = [t["label"] for t in self.terms]
        w = max([len("term")] + [len(t) for t in labels])
        lines = [
            f"logistic — {self.formula}   (n={self.n_obs}, α={self.alpha})",
            f"  {self.model}",
            "",
            "fixed effects (log-odds of a PASS; + = more likely; OR = odds ratio):",
            f"  {'term':<{w}}{'log-odds':>10}{'OR':>10}{'p':>11}     ",
        ]
        for t in self.terms:
            coef = f"{t['coef']:+.3f}" if t.get("coef") is not None else "-"
            if t.get("odds_ratio") is None:
                orr = "-"
            elif t["odds_ratio"] >= 1e4:
                orr = f"{t['odds_ratio']:.0e}"
            else:
                orr = f"{t['odds_ratio']:.2f}"
            p = "-" if t.get("p") is None else f"{t['p']:.4f}"
            lines.append(f"  {t['label']:<{w}}{coef:>10}{orr:>10}{p:>11}   {sig_code(t.get('p'))}")
        withheld = self.converged is False and not _separated(self)  # the notes below say why
        if self.terms and all(t.get("p") is None for t in self.terms) and not withheld:
            lines.append("")
            lines.append("  note: standard errors / p-values unavailable — the model is near-degenerate "
                         "(a factor separates pass from fail); the odds ratios are unstable.")
        elif not withheld:
            lines.append(f"  {SIG_LEGEND}")
        if self.variance_components:
            from cafe.stats.ordinal import _variance_line

            lines.append("")
            lines.append(_variance_line(self.variance_components))
        if self.warnings:
            lines.append("")
            for wn in self.warnings:
                lines.append(f"note: {wn}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        if not self.available:
            return f"Logistic(unavailable: {self.reason})"
        return f"Logistic({len(self.terms)} term(s), n={self.n_obs})"


def check_glmer() -> tuple[bool, str]:
    """Return ``(ok, message)`` for R + ``lme4`` (the GLMM backend) availability."""
    if shutil.which("Rscript") is None:
        return False, "Rscript not found on PATH"
    try:
        proc = subprocess.run(
            ["Rscript", "-e",
             'cat(requireNamespace("lme4", quietly=TRUE) && requireNamespace("jsonlite", quietly=TRUE))'],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Rscript present but failed to run: {exc}"
    if "TRUE" in proc.stdout:
        return True, "R and the 'lme4' package are available."
    return False, "R is installed but the 'lme4' package is missing"


def _display_formula(usable: list[str], order: int) -> str:
    fixed = f"({' + '.join(usable)})^{order}" if order >= 2 and len(usable) >= 2 else " + ".join(usable)
    return f"verdict ~ (1 | input_id) + {fixed}"


def _separated(res: Logistic) -> bool:
    """Whether a factor (near-)perfectly predicts pass/fail: |log-odds| > 10 (OR > ~22000)."""
    return any(t["coef"] is not None and abs(t["coef"]) > 10 for t in res.terms)


def _apply_separation_guard(res: Logistic) -> None:
    """A factor that (near-)perfectly predicts pass/fail makes |log-odds| explode with a
    spuriously tiny p. Suppress the p-values as unreliable (like the CLMM degeneracy note);
    the pass rates in the descriptive layer stand. |log-odds| > 10 is OR > ~22000."""
    if _separated(res):
        for t in res.terms:
            t["p"] = None
            t["significant"] = False


def _apply_convergence_guard(res: Logistic, optimizer: dict[str, Any] | None) -> None:
    """Set ``res.converged`` from glmm.R's ``optimizer`` status, as lme4 itself judges it:
    the optimizer's convergence code (0 = success, ``?lmerControl``) and lme4's derivative
    checks (``?convergence``), which mark a failure with a negative code / "Model failed to
    converge" (glmm.R must silence glmer's warnings, so they surface here). After such a
    failure the Hessian behind the standard errors is meaningless; they can be ~100x too
    small (p ≈ 0 for a factor with no effect). So the p-values are dropped and a note is
    added, as ``fit_clmm`` does; the estimates are kept."""
    code = (optimizer or {}).get("convergence")
    if not isinstance(code, int):  # no status reported: leave converged=None
        return
    lme4_codes = [c for c in optimizer.get("lme4_code") or [] if isinstance(c, int)]
    # first line of each message (lme4 appends a "See ?lme4::convergence" hint), de-duplicated
    raw = [str(m).strip() for m in optimizer.get("lme4_messages") or []]
    msgs = list(dict.fromkeys(m.splitlines()[0] for m in raw if m))
    # lme4 may overwrite a gradient failure's code with a later Hessian check's, so its
    # "failed to converge" wording counts too. A singular (boundary) fit is not a failure.
    res.converged = (code == 0 and not any(c < 0 for c in lme4_codes)
                     and not any("failed to converge" in m for m in msgs))
    if res.converged:
        if optimizer.get("refit"):  # glmm.R refit after the default optimizer's fit failed
            res.warnings.append(
                f"lme4's default optimizer did not converge; refit with {optimizer['refit']}, "
                "which did — the results are from that fit"
            )
        return
    for t in res.terms:
        t["p"] = None
        t["significant"] = False
    why = "; ".join(msgs) if code == 0 else f"optimizer code {code}: {optimizer.get('message')}"
    res.warnings.append(
        f"R's optimizer did not converge (lme4: {why}) — the standard errors of this fit are "
        "not valid, so no p-values are shown; treat the estimates as unreliable and read the "
        "descriptive pass rates instead"
    )


def _fit_glmer(
    df, factors: list[str], *, alpha: float, interactions: int, timeout: int
) -> Logistic | None:
    """Fit the logistic GLMM via R ``lme4::glmer``. Returns a :class:`Logistic`, or
    ``None`` if R produced no usable model (so the caller falls back to statsmodels)."""
    import json
    import os
    import tempfile
    from importlib.resources import files

    from cafe.stats.ordinal import _r_coef_term

    order = max(1, min(interactions, len(factors)))
    keep = ["verdict", "input_id", *factors]
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    try:
        df[keep].to_csv(tmp.name, index=False)
        tmp.close()
        r_script = files("cafe.stats").joinpath("glmm.R")
        try:
            proc = subprocess.run(
                ["Rscript", str(r_script), tmp.name, ",".join(factors), str(order)],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    if not payload.get("available"):
        return None

    res = Logistic(alpha=alpha, available=True, n_obs=payload.get("n_obs", len(df)))
    res.model = "logistic GLMM (binomial, logit link; random intercept: question) [R lme4::glmer]"
    # Show what R fit: glmm.R refits main effects when the interaction model fails, and the
    # requested ^2 formula would make the missing interaction rows read as "estimated".
    used_order = payload.get("used_order", order)
    res.formula = _display_formula(factors, used_order)
    if used_order < order:
        res.warnings.append("interactions not estimable on this data — main effects only")
    for c in payload.get("coefficients", []):
        label, factor, interaction = _r_coef_term(c, factors)
        est = c.get("estimate")
        p = c.get("p")
        res.terms.append({
            "label": label,
            "factor": factor,
            "interaction": interaction,
            "coef": est,
            "odds_ratio": math.exp(est) if est is not None and abs(est) < 30 else None,
            "p": p,
            "significant": p is not None and p < alpha,
        })
    _apply_convergence_guard(res, payload.get("optimizer"))
    _apply_separation_guard(res)
    from cafe.stats.ordinal import _set_random_intercept

    _set_random_intercept(res, payload)  # variance_components (σ_u², ICC) + singular note
    return res


def _fit_statsmodels(df, factors: list[str], *, alpha: float, interactions: int) -> Logistic:
    """Self-contained logistic fallback: GEE (cluster-robust by question), or plain
    logistic when there's one question group / GEE won't fit."""
    res = Logistic(alpha=alpha, n_obs=len(df))
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.genmod.families import Binomial
        from statsmodels.tools.sm_exceptions import PerfectSeparationError
    except ImportError:
        res.reason = "statsmodels not installed (pip install 'cafe[stats]')"
        return res

    # Fit on collision-proof column names (a factor named like a patsy builtin C/Q/I would
    # otherwise shadow the formula function); map terms back to real names below.
    from cafe.stats.inferential import _safe_factor_frame

    dfs, safe, back = _safe_factor_frame(df, factors)
    order = max(1, min(interactions, len(factors)))
    main = " + ".join(f"C(Q('{safe[c]}'))" for c in factors)
    formula = f"verdict ~ ({main})**{order}" if order >= 2 and len(factors) >= 2 else f"verdict ~ {main}"
    res.formula = _display_formula(factors, order)
    have_groups = "input_id" in dfs.columns and dfs["input_id"].nunique() >= 2

    import warnings as _warnings

    fit = None
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        try:
            if not have_groups:
                raise RuntimeError("single question group")
            fit = smf.gee(
                formula, groups="input_id", data=dfs,
                family=Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
            ).fit()
            res.model = "GEE logistic (binomial, logit link; cluster-robust by question)"
        except PerfectSeparationError:
            return _separation_result(res, factors)
        except Exception:  # noqa: BLE001 — GEE not fittable; fall back to plain logistic
            try:
                fit = smf.logit(formula, data=dfs).fit(disp=0)
                res.model = "logistic regression (binomial, logit link)"
                res.warnings.append(
                    "cluster-robust question effect not estimable; plain logistic regression"
                    if have_groups else "only one question group; plain logistic (no question effect)"
                )
            except PerfectSeparationError:
                return _separation_result(res, factors)
            except Exception as exc:  # noqa: BLE001
                res.reason = f"logistic fit failed: {exc}"
                return res

    params, pvals = fit.params, fit.pvalues
    for name in params.index:
        if name in ("Intercept", "const"):
            continue
        label, factor, inter = _parse_statsmodels_term(str(name), back)
        coef = float(params[name])
        p = float(pvals[name]) if pvals[name] == pvals[name] else None
        res.terms.append({
            "label": label,
            "factor": factor,
            "interaction": inter,
            "coef": coef,
            "odds_ratio": math.exp(coef) if abs(coef) < 30 else None,
            "p": p,
            "significant": p is not None and p < alpha,
        })
    res.available = True
    _apply_separation_guard(res)
    return res


def _separation_result(res: Logistic, factors: list[str]) -> Logistic:
    res.available = True
    res.reason = "perfect separation"
    res.warnings.append(
        "a factor perfectly predicts pass/fail (separation) — odds ratios diverge; "
        "read the pass rates in the descriptive layer instead"
    )
    res.terms = [
        {"label": f"{c}", "factor": c, "interaction": False,
         "coef": None, "odds_ratio": None, "p": None, "significant": False}
        for c in factors
    ]
    return res


def fit_logistic(
    ratings: "Ratings", *, alpha: float = 0.05, interactions: int = 2,
    backend: str = "auto", timeout: int = 120,
) -> Logistic:
    """Fit a logistic model of pass/fail (``verdict`` ∈ {0,1}) on the factors.

    ``backend`` selects the engine: ``"glmer"`` (logistic GLMM via R lme4), ``"statsmodels"``
    (self-contained GEE / logistic), or ``"auto"`` (default — GLMM when R+lme4 are present,
    else statsmodels). ``interactions`` is the max interaction order (1 = main effects,
    2 = also two-way); if the GLMM can't fit the interactions, R refits main effects and the
    result's ``formula`` and ``warnings`` say so. Returns a :class:`Logistic`;
    ``available=False`` with a ``reason`` when the rubric isn't binary, there's no variance,
    or no engine is available.

    Returns
    -------
    Logistic
        ``converged`` is lme4's convergence verdict on the GLMM (``None`` for the statsmodels
        backends and when nothing was fit). If the optimizer or lme4's own convergence checks
        failed (``converged=False``), every term's ``p`` is ``None`` and ``significant`` is
        ``False``, because the standard errors of such a fit are not valid; ``warnings``
        says so.
    """
    from cafe.stats._frame import analysis_frame

    res = Logistic(alpha=alpha)

    rubric = getattr(ratings, "rubric", None)
    scale = getattr(getattr(rubric, "scale_type", None), "value", None)
    if scale is not None and scale != "binary":
        res.reason = f"logistic model is for binary scales; this rubric's scale_type is {scale!r}."
        return res

    df = analysis_frame(ratings)
    if df.empty or "verdict" not in df.columns:
        res.reason = "no verdicts to fit"
        return res
    df["verdict"] = df["verdict"].astype(int)
    res.n_obs = len(df)

    values = set(df["verdict"].unique().tolist())
    if not values <= {0, 1}:
        res.reason = "binary logistic needs 0/1 verdicts"
        return res
    if len(values) < 2:
        only = "pass" if 1 in values else "fail"
        res.reason = f"every answer scored the same ({only}) — no variance to model"
        return res

    usable = [
        f for f in ratings.factors
        if f in df.columns and df[f].astype("string").dropna().nunique() >= 2
    ]
    if not usable:
        res.reason = "no factor varies across at least two levels"
        return res

    # Backend selection: prefer the GLMM (glmer) when available.
    want_glmer = backend in ("auto", "glmer")
    fallback_note = None
    if want_glmer:
        ok, msg = check_glmer()
        if ok:
            glmm = _fit_glmer(df, usable, alpha=alpha, interactions=interactions, timeout=timeout)
            if glmm is not None:
                return glmm
            fallback_note = "R GLMM did not converge; used the statsmodels logistic instead"
        elif backend == "glmer":
            res.reason = f"logistic GLMM (R lme4::glmer) unavailable: {msg}"
            return res
        else:  # auto, R/lme4 missing
            fallback_note = f"logistic GLMM (R lme4) unavailable ({msg}); used the statsmodels logistic"

    res = _fit_statsmodels(df, usable, alpha=alpha, interactions=interactions)
    if fallback_note and res.available:
        res.warnings.insert(0, fallback_note)
    return res
