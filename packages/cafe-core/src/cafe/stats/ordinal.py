"""Ordinal cumulative-link mixed model (CLMM) — the statistically correct model for
ordinal judge verdicts.

R has the gold-standard implementation (``ordinal::clmm``); Python has no equal. So
this layer shells out to R via ``Rscript`` and a JSON contract — there is **no
rpy2 build dependency**, the Python package installs cleanly with pip, and R is a
*runtime* prerequisite only for this one layer. If R (or the ``ordinal`` package)
isn't present, ``fit_clmm`` returns a result with ``available=False`` and clear
install instructions, and the rest of CAFE (incl. the Gaussian mixed model in
:mod:`cafe.stats.inferential`) keeps working.

Check your setup with ``cafe doctor``.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cafe.judging.ratings import Ratings

_INSTALL_HINT = (
    "Install R (https://www.r-project.org/, e.g. `apt install r-base`) and the "
    "ordinal package: Rscript -e 'install.packages(\"ordinal\")'."
)


def _term_factor(term: str, factors: list[str]) -> str | None:
    """The factor an R coefficient name starts with. The **longest** matching name wins:
    with factors ``retrieve`` and ``retrieve.top_k``, ``retrieve.top_k2`` belongs to
    ``retrieve.top_k`` (a first-match would credit it to ``retrieve``)."""
    matches = [f for f in factors if term.startswith(f)]
    return max(matches, key=len) if matches else None


def _readable_term(term: str, factors: list[str]) -> str:
    """Turn R's ``retrievekeyword:reranknone`` into ``retrieve=keyword × rerank=none``."""
    parts = []
    for piece in term.split(":"):
        f = _term_factor(piece, factors)
        parts.append(f"{f}={piece[len(f):]}" if f else piece)
    return " × ".join(parts)


def _r_coef_term(c: dict[str, Any], factors: list[str]) -> tuple[str, str | None, bool]:
    """``(label, factor, is_interaction)`` for one coefficient reported by clmm.R / glmm.R.

    The R scripts send each coefficient's ``factors`` and ``levels``. That is exact even when
    a level contains ``:`` (``model="llama3:8b"`` has the R name ``modelllama3:8b``, which
    reads like an interaction) and when R back-quotes a non-syntactic factor name
    (`` `top-k`5 ``). Only if they are missing is R's concatenated name parsed instead."""
    fs, lv = c.get("factors") or [], c.get("levels") or []
    if fs and len(fs) == len(lv):
        return " × ".join(f"{f}={v}" for f, v in zip(fs, lv)), fs[0], len(fs) > 1
    term = str(c.get("term", ""))
    return _readable_term(term, factors), _term_factor(term, factors), ":" in term  # R names interactions a:b


def check_r() -> tuple[bool, str]:
    """Return ``(ok, message)`` describing R + ``ordinal`` availability."""
    if shutil.which("Rscript") is None:
        return False, f"Rscript not found on PATH. {_INSTALL_HINT}"
    try:
        proc = subprocess.run(
            ["Rscript", "-e", 'cat(requireNamespace("ordinal", quietly=TRUE))'],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Rscript present but failed to run: {exc}"
    if "TRUE" in proc.stdout:
        return True, "R and the 'ordinal' package are available."
    return False, f"R is installed but the 'ordinal' package is missing. {_INSTALL_HINT}"


# Residual variance of a logit-link model on its latent scale (the variance of the standard
# logistic distribution, π²/3) — the denominator term of the CLMM / GLMM latent-scale ICC.
_LATENT_LOGIT_VARIANCE = math.pi ** 2 / 3

# Below this random-intercept SD the fit counts as singular (σ_u ≈ 0). lme4 puts a collapsed
# intercept at exactly 0, but clmm stops short of it (around 1e-4), so lme4's own 1e-4
# isSingular tolerance would miss many collapsed CLMM fits.
_SINGULAR_SD = 1e-3


def _set_random_intercept(res: Any, payload: dict[str, Any]) -> None:
    """Fill ``res.variance_components`` from the R payload's ``random_intercept``
    (``VarCorr``: variance + sd of the per-question intercept) and add a note to
    ``res.warnings`` when that intercept collapsed to ~0. Shared by the CLMM and the GLMM."""
    ri = payload.get("random_intercept") or {}
    if ri.get("variance") is None:
        return
    var = float(ri["variance"])
    sd = float(ri["sd"]) if ri.get("sd") is not None else math.sqrt(var)
    res.variance_components = {
        "random_intercept": var,
        "random_intercept_sd": sd,
        "residual": _LATENT_LOGIT_VARIANCE,
        "icc": var / (var + _LATENT_LOGIT_VARIANCE),
    }
    if sd < _SINGULAR_SD:
        res.warnings.append(
            "model was near-singular (random-intercept variance ≈ 0: the questions don't differ "
            "beyond what the factors explain) — a boundary fit, not a failure: the estimates and "
            "p-values are valid and equal those of the same model without a question effect"
        )


def _variance_line(vc: dict[str, float]) -> str:
    """One-line summary of :func:`_set_random_intercept`'s components for ``show()``."""
    return (
        f"variance components:  between-question σ_u² = {vc['random_intercept']:.3f}"
        f" (σ_u = {vc['random_intercept_sd']:.3f}) · latent residual π²/3 = {vc['residual']:.3f}"
        f" · ICC = {vc['icc']:.3f}"
    )


@dataclass
class CLMMResult:
    """Result of the ordinal CLMM (or why it couldn't be fit)."""

    available: bool = False
    reason: str | None = None        # why unavailable (R missing, no variance, ...)
    error: str | None = None         # R-side error message, if any
    alpha: float = 0.05
    n_obs: int | None = None
    formula: str | None = None       # the fitted model (main effects only after a fallback)
    log_lik: float | None = None
    converged: bool | None = None    # R's optimizer status; False -> std_error / z / p are None
    coefficients: list[dict[str, Any]] = field(default_factory=list)  # term, estimate, std_error, z, p, factor, significant
    thresholds: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: The per-question random intercept on the latent logit scale (like R's ``VarCorr(fit)``):
    #: ``{"random_intercept": σ_u², "random_intercept_sd": σ_u, "residual": π²/3,
    #: "icc": σ_u² / (σ_u² + π²/3)}``, or ``None`` when no model was fit.
    variance_components: dict[str, float] | None = None

    @property
    def significant_factors(self) -> list[str]:
        return sorted({c["factor"] for c in self.coefficients if c.get("significant") and c.get("factor")})

    def to_df(self):
        """The fixed-effects table as a tidy DataFrame (term, estimate, std_error, z, p,
        significant) — the data behind :meth:`show`. Needs the ``stats`` extra."""
        import pandas as pd

        cols = ["label", "term", "estimate", "std_error", "z", "p", "factor", "significant"]
        return pd.DataFrame([{c: co.get(c) for c in cols} for co in self.coefficients])

    def show(self) -> str:
        from cafe.stats._format import SIG_LEGEND, sig_code

        if not self.available:
            return f"ordinal CLMM unavailable: {self.reason or self.error}"
        ll = f"{self.log_lik:.1f}" if self.log_lik is not None else "?"
        labels = [c.get("label") or c.get("term", "") for c in self.coefficients]
        w = max([len("term")] + [len(t) for t in labels])
        lines = [
            f"ordinal CLMM — {self.formula}   (n={self.n_obs}, logLik={ll}, α={self.alpha})",
            "",
            "fixed effects (ordinal log-odds of a higher score; + = better):",
            f"  {'term':<{w}}{'estimate':>11}{'p':>11}     ",
        ]
        for c, term in zip(self.coefficients, labels):
            est = f"{c['estimate']:+.3f}" if c.get("estimate") is not None else "-"
            p = "-" if c.get("p") is None else f"{c['p']:.4f}"
            lines.append(f"  {term:<{w}}{est:>11}{p:>11}   {sig_code(c.get('p'))}")
        if self.converged is False:
            lines.append("")  # p-values withheld; the optimizer note below says why
        elif self.coefficients and all(c.get("p") is None for c in self.coefficients):
            lines.append("")
            lines.append("  note: standard errors / p-values unavailable — the model is near-degenerate "
                         "(separation or a ceiling in the scores); the estimates are unstable.")
        else:
            lines.append(f"  {SIG_LEGEND}")
        if self.variance_components:
            lines.append(f"  {_variance_line(self.variance_components)}")
        for w in self.warnings:
            lines.append(f"  note: {w}")
        return "\n".join(lines)


def fit_clmm(
    ratings: "Ratings", *, alpha: float = 0.05, interactions: int = 2, timeout: int = 120
) -> CLMMResult:
    """Fit an ordinal CLMM (``verdict ~ factors + (1|question)``) via R.

    ``interactions`` is the maximum interaction order (1 = main effects, 2 = also two-way,
    …); R falls back to main effects if the interaction model doesn't converge.

    Returns
    -------
    CLMMResult
        Always returned rather than raised: ``available=False`` with a ``reason`` when R /
        ``ordinal`` is missing or there is nothing to fit. Aliased interactions (e.g. a
        fractional factorial below resolution V) are dropped with a note in ``warnings``;
        without statsmodels that check can't run, and ``warnings`` says so instead.

        ``converged`` is R's optimizer status (``None`` when nothing was fit). If the
        optimizer stopped without converging (``converged=False``), every coefficient's
        ``std_error``, ``z`` and ``p`` are ``None`` and ``significant`` is ``False``, because
        the standard errors of such a fit are not valid; ``warnings`` says so.
    """
    import json
    import os
    import tempfile
    from importlib.resources import files

    from cafe.stats._frame import analysis_frame

    res = CLMMResult(alpha=alpha)

    rubric = getattr(ratings, "rubric", None)
    scale = getattr(getattr(rubric, "scale_type", None), "value", None)
    if scale is not None and scale != "ordinal":
        res.reason = f"CLMM is for ordinal scales; this rubric's scale_type is {scale!r}."
        return res

    df = analysis_frame(ratings)  # one row per answer (judge replications averaged)
    if df.empty or "verdict" not in df.columns:
        res.reason = "no verdicts to fit"
        return res
    df["verdict"] = df["verdict"].astype(int)

    factors = [
        f for f in ratings.factors
        if f in df.columns and df[f].astype("string").dropna().nunique() >= 2
    ]
    if not factors:
        res.reason = "no factor varies across at least two levels"
        return res

    ok, msg = check_r()
    if not ok:
        res.reason = msg
        return res

    keep = ["verdict", "input_id", *factors]
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    try:
        df[keep].to_csv(tmp.name, index=False)
        tmp.close()
        r_script = files("cafe.stats").joinpath("clmm.R")
        order = max(1, min(interactions, len(factors)))
        # Cap the order to what the design can estimate: an aliased fractional factorial is
        # rank-deficient above main effects, so its 2FIs can't be separated (matches the
        # Gaussian layer, so the two reports agree).
        from cafe.stats.inferential import _design_rank_deficient

        aliased = _design_rank_deficient(df, factors, order) if order >= 2 else False
        if aliased:
            order = 1
            res.warnings.append(
                "interactions are aliased in this design (e.g. a fractional factorial below "
                "resolution V) — they can't be separated, so only main effects are fit"
            )
        elif aliased is None:
            res.warnings.append(
                "statsmodels is not installed, so the design wasn't checked for aliased "
                "interactions; for a fractional factorial below resolution V, read the "
                "interaction terms as confounded (or pass interactions=1)"
            )
        try:
            proc = subprocess.run(
                ["Rscript", str(r_script), tmp.name, ",".join(factors), str(order)],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            res.reason = f"R timed out after {timeout}s"
            return res
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    if proc.returncode != 0:
        res.reason = f"R exited {proc.returncode}: {proc.stderr.strip()[:200]}"
        return res
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        res.reason = f"R produced invalid JSON: {proc.stdout[:200]!r}"
        return res

    if not payload.get("available"):
        res.error = res.reason = payload.get("error", "R reported the model unavailable")
        return res

    res.available = True
    res.n_obs = payload.get("n_obs")
    res.formula = payload.get("formula")
    if payload.get("used_order", order) < order:  # clmm.R refit main effects only
        res.warnings.append("interactions not estimable on this data — main effects only")
    res.log_lik = payload.get("logLik")
    res.thresholds = payload.get("thresholds", [])
    _set_random_intercept(res, payload)  # variance_components (σ_u², ICC) + singular note
    coeffs = payload.get("coefficients", [])
    for c in coeffs:
        # "retrieve=keyword × rerank=none", its first factor, and whether it's an interaction
        c["label"], c["factor"], c["interaction"] = _r_coef_term(c, factors)
        c["significant"] = c.get("p") is not None and c["p"] < alpha
    res.coefficients = coeffs
    _apply_convergence_guard(res, payload.get("optimizer"))
    return res


def _apply_convergence_guard(res: CLMMResult, optimizer: dict[str, Any] | None) -> None:
    """Set ``res.converged`` from clmm.R's ``optimizer`` status (``nlminb``'s convergence
    code: 0 = converged). ``ordinal::clmm`` doesn't check it itself. A fit that stopped
    without converging has a meaningless Hessian, so its standard errors can be ~100x too
    small (p ≈ 0 for a factor with no effect). Its std_error / z / p are dropped and a note
    is added (like ``fit_logistic`` under separation); the estimates are kept."""
    code = (optimizer or {}).get("convergence")
    if not isinstance(code, int):  # no status reported: leave converged=None
        return
    res.converged = code == 0
    if res.converged:
        return
    for c in res.coefficients:
        c["std_error"] = c["z"] = c["p"] = None
        c["significant"] = False
    msg = optimizer.get("message") or f"code {code}"
    res.warnings.append(
        f"R's optimizer did not converge (nlminb: {msg}) — the standard errors of this fit "
        "are not valid, so no p-values are shown; treat the estimates as unreliable and "
        "read the linear layer's tests and effect sizes instead"
    )
