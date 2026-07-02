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


def _readable_term(term: str, factors: list[str]) -> str:
    """Turn R's ``retrievekeyword:reranknone`` into ``retrieve=keyword × rerank=none``."""
    parts = []
    for piece in term.split(":"):
        matches = [f for f in factors if piece.startswith(f)]
        if matches:
            f = max(matches, key=len)  # longest prefix wins (retrieve.top_k vs retrieve)
            parts.append(f"{f}={piece[len(f):]}")
        else:
            parts.append(piece)
    return " × ".join(parts)


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


@dataclass
class CLMMResult:
    """Result of the ordinal CLMM (or why it couldn't be fit)."""

    available: bool = False
    reason: str | None = None        # why unavailable (R missing, no variance, ...)
    error: str | None = None         # R-side error message, if any
    alpha: float = 0.05
    n_obs: int | None = None
    formula: str | None = None
    log_lik: float | None = None
    coefficients: list[dict[str, Any]] = field(default_factory=list)  # term, estimate, std_error, z, p, factor, significant
    thresholds: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def significant_factors(self) -> list[str]:
        return sorted({c["factor"] for c in self.coefficients if c.get("significant") and c.get("factor")})

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
        if self.coefficients and all(c.get("p") is None for c in self.coefficients):
            lines.append("")
            lines.append("  note: standard errors / p-values unavailable — the model is near-degenerate "
                         "(separation or a ceiling in the scores); the estimates are unstable.")
        else:
            lines.append(f"  {SIG_LEGEND}")
        for w in self.warnings:
            lines.append(f"  note: {w}")
        return "\n".join(lines)


def fit_clmm(
    ratings: "Ratings", *, alpha: float = 0.05, interactions: int = 2, timeout: int = 120
) -> CLMMResult:
    """Fit an ordinal CLMM (``verdict ~ factors + (1|question)``) via R.

    ``interactions`` is the maximum interaction order (1 = main effects, 2 = also two-way,
    …); R falls back to main effects if the interaction model doesn't converge.
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

        if order >= 2 and _design_rank_deficient(df, factors, order):
            order = 1
            res.warnings.append(
                "interactions are aliased in this design (e.g. a fractional factorial below "
                "resolution V) — they can't be separated, so only main effects are fit"
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
    res.log_lik = payload.get("logLik")
    res.thresholds = payload.get("thresholds", [])
    coeffs = payload.get("coefficients", [])
    for c in coeffs:
        term = str(c.get("term", ""))
        c["interaction"] = ":" in term  # R names interaction coefficients a:b
        c["factor"] = next((f for f in factors if term.startswith(f)), None)
        c["label"] = _readable_term(term, factors)  # "retrieve=keyword × rerank=none"
        c["significant"] = c.get("p") is not None and c["p"] < alpha
    res.coefficients = coeffs
    return res
