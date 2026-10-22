"""Statistics layers, exercised on small fabricated Ratings (no LLM, no network)."""

import shutil

import pytest

from cafe import ANSWER_QUALITY_1_5, attribute, fit_clmm, fit_effects
from cafe.judging.ratings import Rating, Ratings


def _ratings(level_to_score: dict[str, int], n_questions: int = 5, reps: int = 2) -> Ratings:
    """Build ratings for one factor 'method' where each level maps to a fixed score."""
    items = []
    for q in range(n_questions):
        for level, score in level_to_score.items():
            for rep in range(reps):
                items.append(
                    Rating(
                        obs_key=f"{level}-{q}-{rep}",
                        config={"method": level},
                        input_id=f"q{q}",
                        rep=rep,
                        judge_rep=0,
                        value=score,
                        value_numeric=score,
                    )
                )
    return Ratings(rubric=ANSWER_QUALITY_1_5, judge_model="fake", factors=["method"], items=items)


def test_attribute_recovers_means():
    attr = attribute(_ratings({"good": 5, "bad": 2}))
    by_level = {m["level"]: m["mean"] for m in attr.factor_marginals}
    assert by_level["good"] == pytest.approx(5.0)
    assert by_level["bad"] == pytest.approx(2.0)
    assert attr.best_config["config"] == {"method": "good"}


def test_attribute_no_factors_reports_overall_mean_only():
    items = [Rating(obs_key=f"k{i}", config={}, input_id=f"q{i}", rep=0, judge_rep=0,
                    value=v, value_numeric=v) for i, v in enumerate([5, 4, 5, 3, 4])]
    attr = attribute(Ratings(rubric=ANSWER_QUALITY_1_5, judge_model="fake",
                             factors=[], items=items))
    assert attr.factors == []
    assert attr.overall_mean == pytest.approx(4.2)
    assert attr.config_means == [] and attr.factor_marginals == []
    text = attr.show()
    assert "overall mean quality: 4.20" in text
    assert "no factors varied" in text
    assert "per-configuration" not in text  # empty sections are suppressed


def test_fit_effects_detects_real_effect():
    eff = fit_effects(_ratings({"good": 5, "bad": 2}))
    assert "method" in eff.significant_factors
    term = next(t for t in eff.terms if t["factor"] == "method")
    assert term["p"] is not None and term["p"] < 0.05
    assert term["partial_eta_sq"] > 0.5  # the factor explains most of the variance


def test_fit_effects_null_when_no_difference():
    eff = fit_effects(_ratings({"a": 4, "b": 4}))
    assert eff.significant_factors == []


def _exact_two_factor_ratings(score: dict[tuple[str, str], int], b_levels: int = 2) -> Ratings:
    """Factors A × B where every answer of a configuration gets the same score on every
    question — the model fits exactly (zero residual variance)."""
    items = []
    for q in range(8):
        for a in ("a1", "a2"):
            for b in (f"b{i}" for i in range(1, b_levels + 1)):
                s = score[(a, b)]
                items.append(Rating(obs_key=f"{a}{b}{q}", config={"A": a, "B": b},
                                    input_id=f"q{q}", rep=0, judge_rep=0, value=s, value_numeric=s))
    return Ratings(rubric=ANSWER_QUALITY_1_5, judge_model="fake", factors=["A", "B"], items=items)


def test_fit_effects_exact_fit_separates_real_from_null_terms():
    """Zero residual variance: A decides the score, B has exactly no effect. A must be
    significant (F=inf, p=0, η²=1); B and A×B have zero SS *and* zero residual SS (0/0) and
    must not be — before, their F was a ratio of rounding noise and B could reach p<0.001."""
    import math

    score = {(a, f"b{i}"): (3 if a == "a1" else 2) for a in ("a1", "a2") for i in (1, 2, 3)}
    r = _exact_two_factor_ratings(score, b_levels=3)
    for order in (1, 2):
        eff = fit_effects(r, interactions=order)
        terms = {t["factor"]: t for t in eff.terms}
        assert eff.significant_factors == ["A"]
        assert math.isinf(terms["A"]["F"]) and terms["A"]["p"] == 0.0
        assert terms["A"]["partial_eta_sq"] == 1.0
        null_terms = ["B"] + (["A × B"] if order == 2 else [])
        for name in null_terms:
            assert terms[name]["p"] is None and terms[name]["F"] is None
            assert terms[name]["partial_eta_sq"] is None and not terms[name]["significant"]
        assert "exact fit" in eff.model
        assert any("zero residual variance" in w for w in eff.warnings)
        assert not any("near-singular" in w for w in eff.warnings)  # replaced by the note above


def _two_factor_ratings(cells: dict[tuple[str, str, str], int]) -> Ratings:
    """Ratings from {(question, A, B): score}."""
    items = [Rating(obs_key=f"{q}{a}{b}", config={"A": a, "B": b}, input_id=q, rep=0, judge_rep=0,
                    value=v, value_numeric=v) for (q, a, b), v in cells.items()]
    return Ratings(rubric=ANSWER_QUALITY_1_5, judge_model="fake", factors=["A", "B"], items=items)


def test_fit_effects_drops_the_question_block_when_a_factor_is_confounded_with_it():
    """B's levels split the questions (q0-q4 only saw b1, q5-q9 only b2): blocking by question
    would absorb B, so the ANOVA must not block (and say why) rather than report a bogus F."""
    import random

    rng = random.Random(3)
    cells = {(f"q{q}", a, "b1" if q < 5 else "b2"): rng.choice([2, 3, 4]) + (a == "a2")
             for q in range(10) for a in ("a1", "a2")}
    eff = fit_effects(_two_factor_ratings(cells), interactions=1)
    assert "blocked" not in eff.model and eff.formula == "verdict ~ A + B"
    assert any("confounded with the questions" in w for w in eff.warnings)
    assert {t["factor"] for t in eff.terms} == {"A", "B"}


def test_fit_effects_tiny_design_is_not_declared_exact_by_the_question_block():
    """2 questions x 1 answer: question + A explain every score, but A alone doesn't — the
    blocked fit is exact by construction, so the test falls back to the unblocked ANOVA
    instead of reporting A as F=inf, p=0."""
    import math

    cells = {("q0", "a1", "b1"): 2, ("q0", "a2", "b1"): 3, ("q1", "a1", "b1"): 4,
             ("q1", "a2", "b1"): 5, ("q0", "a1", "b2"): 2, ("q0", "a2", "b2"): 3,
             ("q1", "a1", "b2"): 4, ("q1", "a2", "b2"): 5}
    eff = fit_effects(_two_factor_ratings(cells), interactions=1)
    a = next(t for t in eff.terms if t["factor"] == "A")
    assert a["F"] is not None and not math.isinf(a["F"])
    assert "exact fit" not in eff.model and "blocked" not in eff.model
    assert any("too few answers per question" in w for w in eff.warnings)


def test_fit_effects_exact_fit_pure_interaction():
    """An exact fit where only the a2·b2 cell is bad: every Type-II term has a real effect."""
    import math

    score = {("a1", "b1"): 5, ("a1", "b2"): 5, ("a2", "b1"): 5, ("a2", "b2"): 2}
    eff = fit_effects(_exact_two_factor_ratings(score), interactions=2)
    terms = {t["factor"]: t for t in eff.terms}
    assert set(terms) == {"A", "B", "A × B"}
    for t in terms.values():
        assert t["significant"] and t["p"] == 0.0
        assert math.isinf(t["F"]) and t["partial_eta_sq"] == 1.0


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_fit_clmm_runs_when_r_present():
    res = fit_clmm(_ratings({"good": 5, "bad": 2}))
    # Either a clean fit, or an honest reason (e.g. 'ordinal' package missing).
    if res.available:
        assert res.coefficients and res.n_obs == 20
    else:
        assert res.reason


def test_fit_clmm_reason_when_no_variance():
    res = fit_clmm(_ratings({"a": 4}))  # single level -> nothing to model
    assert not res.available and res.reason


def _two_factor_interaction_ratings() -> Ratings:
    """Two factors where only the a2·b2 cell is bad — a pure interaction."""
    score = {("a1", "b1"): 5, ("a1", "b2"): 5, ("a2", "b1"): 5, ("a2", "b2"): 2}
    items = []
    for q in range(8):
        for a in ("a1", "a2"):
            for b in ("b1", "b2"):
                s = max(1, score[(a, b)] - (q % 2))  # mild noise so it fits
                items.append(Rating(obs_key=f"{a}{b}{q}", config={"A": a, "B": b},
                                    input_id=f"q{q}", rep=0, judge_rep=0, value=s, value_numeric=s))
    return Ratings(rubric=ANSWER_QUALITY_1_5, judge_model="x", factors=["A", "B"], items=items)


def test_aliasing_check_without_statsmodels_returns_unknown(monkeypatch):
    """The aliasing check fit_clmm runs must not raise ImportError where statsmodels is
    missing (pandas-only installs are supported); it reports 'unknown' (None) instead."""
    import sys

    import pandas as pd

    from cafe.stats.inferential import _design_rank_deficient

    monkeypatch.setitem(sys.modules, "statsmodels.formula.api", None)  # import -> ImportError
    df = pd.DataFrame({"verdict": [1.0, 2.0, 3.0, 4.0],
                       "a": ["x", "x", "y", "y"], "b": ["p", "q", "p", "q"]})
    assert _design_rank_deficient(df, ["a", "b"], order=2) is None
    assert _design_rank_deficient(df, ["a", "b"], order=1) is False  # nothing to check


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_fit_clmm_without_statsmodels_returns_result(monkeypatch):
    """Two factors + the default interactions=2 reach the aliasing check; without statsmodels
    fit_clmm must still return a CLMMResult (and say the check was skipped), not raise."""
    import sys

    monkeypatch.setitem(sys.modules, "statsmodels.formula.api", None)
    res = fit_clmm(_two_factor_interaction_ratings())
    assert res.available or res.reason
    if res.available:
        assert any("statsmodels is not installed" in w for w in res.warnings)


def test_interactions_detected_and_toggleable():
    r = _two_factor_interaction_ratings()
    two_way = {t["factor"]: t for t in fit_effects(r, interactions=2).terms}
    assert "A × B" in two_way and two_way["A × B"]["interaction"] and two_way["A × B"]["significant"]
    main_only = {t["factor"] for t in fit_effects(r, interactions=1).terms}
    assert "A × B" not in main_only and {"A", "B"} <= main_only


def test_interaction_fallback_residuals_come_from_reported_model():
    """One usable answer per cell makes the two-way model saturated (0 residual df), so its
    ANOVA fails and fit_effects falls back to main effects. The residuals/fitted must then
    describe that main-effects model, not the abandoned (exactly fitting) interaction model."""
    cells = {("a1", "b1"): ("q0", 5), ("a1", "b2"): ("q1", 4),
             ("a2", "b1"): ("q2", 3), ("a2", "b2"): ("q0", 1)}
    items = [Rating(obs_key=f"{a}{b}", config={"A": a, "B": b}, input_id=q, rep=0, judge_rep=0,
                    value=s, value_numeric=s) for (a, b), (q, s) in cells.items()]
    r = Ratings(rubric=ANSWER_QUALITY_1_5, judge_model="fake", factors=["A", "B"], items=items)

    eff = fit_effects(r, interactions=2)
    assert any("main effects only" in w for w in eff.warnings)          # the fallback path
    assert not any(t.get("interaction") for t in eff.terms)
    # main-effects fit of 5,4,3,1: fitted 5.25,3.75,2.75,1.25 -> residuals ±0.25 (SSR 0.25);
    # the saturated interaction fit would leave all-zero residuals
    assert sorted(eff.fitted) == pytest.approx([1.25, 2.75, 3.75, 5.25])
    assert sum(e * e for e in eff.residuals) == pytest.approx(0.25)
    assert sorted(f + e for f, e in zip(eff.fitted, eff.residuals)) == pytest.approx([1, 3, 4, 5])


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_clmm_accepts_interaction_order():
    res = fit_clmm(_two_factor_interaction_ratings(), interactions=2)
    # Either a fit (possibly with an interaction coefficient) or an honest reason.
    assert res.available or res.reason


def test_glmer_interaction_fallback_reports_the_fitted_formula():
    """On this data the A×B glmer fails (PIRLS / VtV errors) and glmm.R refits main effects;
    the result must show the main-effects formula and say interactions weren't fit."""
    from cafe import fit_logistic
    from cafe.rubrics import CORRECT_PASS_FAIL
    from cafe.stats.logistic import check_glmer

    if not check_glmer()[0]:
        pytest.skip("R + lme4 not installed")
    cells = [(a, b) for a in ("a1", "a2", "a3") for b in ("b1", "b2", "b3")]
    passes = {"q0": "101001111", "q1": "101101111", "q2": "100001011", "q3": "100101011"}
    items = [Rating(obs_key=f"{q}{a}{b}", config={"A": a, "B": b}, input_id=q, rep=0,
                    judge_rep=0, value=int(v), value_numeric=int(v))
             for q, row in passes.items() for (a, b), v in zip(cells, row)]
    r = Ratings(rubric=CORRECT_PASS_FAIL, judge_model="x", factors=["A", "B"], items=items)
    log = fit_logistic(r, interactions=2, backend="glmer")
    assert log.available and "glmer" in log.model
    assert not any(t["interaction"] for t in log.terms)
    assert log.formula == "verdict ~ (1 | input_id) + A + B"
    assert any("main effects only" in w for w in log.warnings)
    assert "main effects only" in log.show()


def test_clmm_interaction_fallback_is_reported(monkeypatch):
    """clmm.R reports the order it actually fit; a fallback must surface as a warning."""
    import json
    import subprocess

    from cafe.stats import ordinal

    payload = {"available": True, "n_obs": 32, "used_order": 1, "logLik": -30.0,
               "formula": "verdict ~ (1 | input_id) + A + B", "thresholds": [],
               "coefficients": [{"term": "Aa2", "estimate": -1.0, "std_error": 0.5, "z": -2.0,
                                 "p": 0.04}]}
    monkeypatch.setattr(ordinal, "check_r", lambda: (True, "ok"))
    monkeypatch.setattr(ordinal.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, stdout=json.dumps(payload), stderr=""))
    res = fit_clmm(_two_factor_interaction_ratings(), interactions=2)
    assert res.available and res.formula == "verdict ~ (1 | input_id) + A + B"
    assert any("main effects only" in w for w in res.warnings)


def test_readable_clmm_term():
    from cafe.stats.ordinal import _readable_term

    fs = ["retrieve", "retrieve.top_k", "rerank"]
    assert _readable_term("retrievekeyword", fs) == "retrieve=keyword"
    assert _readable_term("retrieve.top_k2", fs) == "retrieve.top_k=2"   # longest-prefix wins
    assert _readable_term("retrievekeyword:reranknone", fs) == "retrieve=keyword × rerank=none"


def test_r_term_factor_is_longest_matching_name():
    from cafe.stats.ordinal import _term_factor

    fs = ["retrieve", "retrieve.top_k", "rerank"]
    assert _term_factor("retrieve.top_k2", fs) == "retrieve.top_k"   # not "retrieve"
    assert _term_factor("retrievekeyword", fs) == "retrieve"
    assert _term_factor("retrieve.top_k2:reranknone", fs) == "retrieve.top_k"
    assert _term_factor("(Intercept)", fs) is None


def _prefix_named_factor_ratings(binary: bool) -> Ratings:
    """Factors ``retrieve`` and ``retrieve.top_k`` (a stage and its parameter factor, so one
    name is a prefix of the other). Only ``retrieve.top_k`` moves the score; a per-question
    difficulty gives the random intercept something to fit."""
    import random

    from cafe.rubrics import CORRECT_PASS_FAIL

    rng = random.Random(3)
    items = []
    for q in range(30):
        hard = rng.gauss(0, 0.7)
        for retrieve in ("bm25", "dense"):
            for top_k in (2, 4):
                eta = (1.5 if top_k == 4 else -1.0) + hard + rng.gauss(0, 1)
                v = int(eta > 0) if binary else max(1, min(5, round(3 + eta)))
                items.append(Rating(obs_key=f"{q}{retrieve}{top_k}",
                                    config={"retrieve": retrieve, "retrieve.top_k": top_k},
                                    input_id=f"q{q}", rep=0, judge_rep=0, value=v, value_numeric=v))
    return Ratings(rubric=CORRECT_PASS_FAIL if binary else ANSWER_QUALITY_1_5, judge_model="x",
                   factors=["retrieve", "retrieve.top_k"], items=items)


def test_clmm_credits_coefficient_to_longest_factor_name():
    from cafe.stats.ordinal import check_r

    if not check_r()[0]:
        pytest.skip("R + ordinal not installed")
    clmm = fit_clmm(_prefix_named_factor_ratings(binary=False), interactions=1)
    assert clmm.available, clmm.reason
    top_k = next(c for c in clmm.coefficients if c["term"] == "retrieve.top_k4")
    assert top_k["factor"] == "retrieve.top_k" and top_k["label"] == "retrieve.top_k=4"
    assert clmm.significant_factors == ["retrieve.top_k"]


def test_glmer_credits_coefficient_to_longest_factor_name():
    from cafe import fit_logistic
    from cafe.stats.logistic import check_glmer

    if not check_glmer()[0]:
        pytest.skip("R + lme4 not installed")
    log = fit_logistic(_prefix_named_factor_ratings(binary=True), interactions=1, backend="glmer")
    assert log.available and "glmer" in log.model
    top_k = next(t for t in log.terms if t["label"] == "retrieve.top_k=4")
    assert top_k["factor"] == "retrieve.top_k"
    assert log.significant_factors == ["retrieve.top_k"]


def test_report_has_all_three_layers():
    from cafe.evaluation import Evaluation
    from cafe.execution.results import Results

    r = _ratings({"good": 5, "bad": 2})
    ev = Evaluation(study_name="t", answers=Results(study_name="t", factors=["method"]),
                    ratings=r, attribution=attribute(r))
    rep = ev.report()
    assert "DESCRIPTIVE" in rep and "INFERENTIAL" in rep and "ORDINAL" in rep
    assert "Cohen's d" in rep
    assert "pipeline:" in rep
    # the mean is reachable as a first-class attribute (no manual reduction needed)
    assert ev.overall_mean == pytest.approx(3.5)
    assert Evaluation(study_name="t", answers=Results(study_name="t", factors=[])).overall_mean is None


def _canned_clmm(monkeypatch, optimizer):
    """fit_clmm on a canned clmm.R payload whose fit reports ``optimizer`` (no R needed)."""
    import json
    import subprocess

    from cafe.stats import ordinal

    payload = {"available": True, "n_obs": 20, "logLik": -10.0, "thresholds": [],
               "formula": "verdict ~ (1 | input_id) + method", "optimizer": optimizer,
               "coefficients": [{"term": "methodgood", "estimate": 0.24, "std_error": 0.0028,
                                 "z": 84.7, "p": 0.0}]}
    monkeypatch.setattr(ordinal, "check_r", lambda: (True, "ok"))
    monkeypatch.setattr(ordinal.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, stdout=json.dumps(payload), stderr=""))
    return fit_clmm(_ratings({"good": 5, "bad": 2}))


def test_clmm_parses_per_factor_likelihood_ratio_tests(monkeypatch):
    """clmm.R's per-term LR tests become factor_tests: interactions labelled "A × B",
    significance from p, a note for a term whose reduced model didn't converge."""
    import json
    import subprocess

    from cafe.stats import ordinal

    payload = {"available": True, "n_obs": 20, "logLik": -10.0, "thresholds": [],
               "formula": "verdict ~ (1 | input_id) + method", "coefficients": [],
               "optimizer": {"convergence": 0, "message": "relative convergence (4)"},
               "factor_tests": [
                   {"term": "method", "factors": ["method"], "df": 1, "chisq": 12.5, "p": 0.0004,
                    "converged": True},
                   {"term": "method:other", "factors": ["method", "other"], "df": 2,
                    "chisq": None, "p": None, "converged": False}]}
    sent = []
    monkeypatch.setattr(ordinal, "check_r", lambda: (True, "ok"))

    def fake_run(cmd, **kwargs):
        sent.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(ordinal.subprocess, "run", fake_run)
    res = fit_clmm(_ratings({"good": 5, "bad": 2}))
    assert sent[-1][-1] == "1"                              # LR tests requested by default
    tests = {t["term"]: t for t in res.factor_tests}
    assert tests["method"]["significant"] and tests["method"]["df"] == 1
    assert tests["method × other"]["interaction"] and tests["method × other"]["p"] is None
    assert res.significant_factors == ["method"]
    assert any("no likelihood-ratio test for method × other" in w for w in res.warnings)
    assert "per-factor tests (likelihood ratio" in res.show()
    fit_clmm(_ratings({"good": 5, "bad": 2}), lr_tests=False)
    assert sent[-1][-1] == "0"


def test_clmm_likelihood_ratio_tests_find_the_real_factor():
    from cafe.stats.ordinal import check_r

    if not check_r()[0]:
        pytest.skip("R + ordinal not installed")
    r = _prefix_named_factor_ratings(binary=False)            # top_k matters, retrieve doesn't
    clmm = fit_clmm(r, interactions=1)
    tests = {t["term"]: t for t in clmm.factor_tests}
    assert set(tests) == {"retrieve", "retrieve.top_k"}
    assert tests["retrieve.top_k"]["significant"] and not tests["retrieve"]["significant"]
    assert tests["retrieve.top_k"]["df"] == 1 and tests["retrieve.top_k"]["chisq"] > 0
    assert fit_clmm(r, interactions=1, lr_tests=False).factor_tests == []


def test_single_factor_studies_get_their_likelihood_ratio_test():
    """The reduced model of a one-factor study is intercept-only; it must still be fitted
    (it used to get the cache key "", which R can't look up -> no test + a false note)."""
    from cafe import fit_logistic
    from cafe.stats.logistic import check_glmer
    from cafe.stats.ordinal import check_r

    if not check_r()[0]:
        pytest.skip("R + ordinal not installed")
    clmm = fit_clmm(_ratings({"good": 5, "bad": 2}), interactions=1)
    assert len(clmm.factor_tests) == 1 and clmm.factor_tests[0]["p"] is not None
    assert not any("didn't converge" in w for w in clmm.warnings)
    if check_glmer()[0]:
        log = fit_logistic(_binary_ratings(good_vals=[1, 1, 1, 0] * 3, bad_vals=[0, 0, 1, 0] * 3),
                           backend="glmer")
        assert len(log.factor_tests) == 1 and log.factor_tests[0]["p"] is not None


def test_clmm_withholds_pvalues_when_optimizer_did_not_converge(monkeypatch):
    """clmm doesn't check nlminb's status; after a false convergence its standard errors
    are ~100x too small (p ≈ 0). Such a fit must not report p-values."""
    res = _canned_clmm(monkeypatch, {"convergence": 1, "message": "false convergence (8)"})
    assert res.available and res.converged is False
    c = res.coefficients[0]
    assert c["p"] is None and c["z"] is None and c["std_error"] is None
    assert c["significant"] is False and res.significant_factors == []
    assert c["estimate"] == pytest.approx(0.24)
    assert any("did not converge" in w and "false convergence (8)" in w for w in res.warnings)
    text = res.show()
    assert "did not converge" in text and "near-degenerate" not in text and "Signif." not in text

    ok = _canned_clmm(monkeypatch, {"convergence": 0, "message": "relative convergence (4)"})
    assert ok.converged is True and ok.coefficients[0]["p"] == 0.0 and not ok.warnings


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_clmm_no_spurious_significance_from_unconverged_fit():
    """30 questions; `prompt` has no effect. On this data clmm's optimizer stops with
    "false convergence (8)" (R 4.5 / ordinal 2026.7) and reports SE 0.003 instead of ~0.33,
    so p ≈ 0. Whatever the optimizer does, `prompt` must not come out significant."""
    import random

    rng = random.Random(5)
    items = []
    for q in range(30):
        for m in ("small", "large"):
            for p in ("plain", "cot"):
                v = max(1, min(5, round(3 + (1.0 if m == "large" else -0.5) + rng.gauss(0, 1))))
                items.append(Rating(obs_key=f"{q}{m}{p}", config={"model": m, "prompt": p},
                                    input_id=f"q{q}", rep=0, judge_rep=0, value=v, value_numeric=v))
    r = Ratings(rubric=ANSWER_QUALITY_1_5, judge_model="x", factors=["model", "prompt"], items=items)
    res = fit_clmm(r, interactions=1)
    if not res.available:
        pytest.skip(res.reason or "CLMM unavailable")
    assert res.converged is not None                     # clmm.R reports the optimizer status
    prompt = next(c for c in res.coefficients if c["factor"] == "prompt")
    assert "prompt" not in res.significant_factors
    if res.converged is False:
        assert prompt["p"] is None and any("did not converge" in w for w in res.warnings)
    else:
        assert prompt["std_error"] > 0.1


def _binary_ratings(good_vals, bad_vals):
    from cafe.rubrics import CORRECT_PASS_FAIL

    items = []
    for iid, (g, b) in enumerate(zip(good_vals, bad_vals)):
        for method, sc in (("good", g), ("bad", b)):
            items.append(Rating(obs_key=f"{method}-{iid}", config={"method": method},
                                input_id=f"q{iid}", rep=0, judge_rep=0, value=sc, value_numeric=sc))
    return Ratings(rubric=CORRECT_PASS_FAIL, judge_model="fake", factors=["method"], items=items)


def test_fractional_design_caps_interactions_to_estimable():
    """A resolution-IV fractional factorial aliases its 2-factor interactions, so neither
    the Gaussian nor the ordinal layer may fit them — they cap to main effects and warn."""
    import random

    import cafe
    from cafe import fit_clmm, fit_effects
    from cafe.design.fractional import fractional_factorial_design

    names = ["model", "sabotage", "verbosity", "hedge"]   # avoid patsy-reserved names (C, Q, I)
    facs = [cafe.Factor(n, [0, 1]) for n in names]
    design = fractional_factorial_design(facs)   # 2^(4-1)
    assert design.resolution == 4
    random.seed(1)
    items = []
    for ci, cfg in enumerate(design.configs):
        for q in range(4):
            sc = max(1, min(5, (5 if cfg["model"] == 1 else 2) + random.choice([-1, 0, 1])))
            items.append(Rating(obs_key=f"{ci}-{q}", config=cfg, input_id=f"q{q}",
                                rep=0, judge_rep=0, value=sc, value_numeric=sc))
    r = Ratings(rubric=ANSWER_QUALITY_1_5, judge_model="fake", factors=names, items=items)

    eff = fit_effects(r, interactions=2)
    assert not any(t.get("interaction") for t in eff.terms)          # no 2FI terms fit
    assert any("aliased" in w for w in eff.warnings)

    clmm = fit_clmm(r, interactions=2)
    if clmm.available:                                               # R present
        assert not any(c.get("interaction") for c in clmm.coefficients)
        assert any("aliased" in w for w in clmm.warnings)


def _uninstall(monkeypatch, *packages):
    """Make importing these packages (and any submodule) fail, as if not installed."""
    import sys

    names = {m for m in sys.modules if m.split(".")[0] in packages}
    names |= {"scipy.stats", "statsmodels.api", "statsmodels.formula.api", *packages}
    for name in names:
        if name.split(".")[0] in packages:
            monkeypatch.setitem(sys.modules, name, None)   # import -> ImportError


def test_fit_effects_without_statsmodels_or_scipy_returns_result(monkeypatch):
    """pandas without the stats extra (e.g. cafe-core[llm,notebooks]): fit_effects' one-way
    fallback must not raise ImportError for scipy; it reports no F / p and says why."""
    from cafe.evaluation import Evaluation
    from cafe.execution.results import Results

    scores = {"good": [5, 4, 5, 5, 4, 5, 4, 5], "bad": [2, 1, 2, 3, 2, 1, 2, 2]}
    items = [Rating(obs_key=f"{lvl}-{q}", config={"method": lvl}, input_id=f"q{q}", rep=0,
                    judge_rep=0, value=s, value_numeric=s)
             for lvl, ss in scores.items() for q, s in enumerate(ss)]
    r = Ratings(rubric=ANSWER_QUALITY_1_5, judge_model="fake", factors=["method"], items=items)
    _uninstall(monkeypatch, "statsmodels")          # scipy present: the fallback has p-values
    term = fit_effects(r).terms[0]
    assert term["p"] is not None and term["significant"]

    _uninstall(monkeypatch, "statsmodels", "scipy")
    eff = fit_effects(r)
    assert eff.model == "one-way ANOVA (statsmodels unavailable)"
    term = eff.terms[0]
    assert term["factor"] == "method" and term["F"] is None and term["p"] is None
    assert term["significant"] is False and term["partial_eta_sq"] > 0.5
    assert eff.pairwise_d and any("scipy is not installed" in w for w in eff.warnings)
    assert "scipy is not installed" in eff.show()
    ev = Evaluation(study_name="t", answers=Results(study_name="t", factors=["method"]),
                    ratings=r, attribution=attribute(r))
    assert "INFERENTIAL" in ev.report()             # was: ModuleNotFoundError (scipy)


def test_factor_named_like_patsy_builtin_does_not_crash():
    """A factor literally named 'C' or 'Q' collides with patsy's formula builtins; the stats
    layer must fit on safe internal names and still report the real names (screening designs
    often use letter names)."""
    import random

    from cafe import fit_effects

    random.seed(0)
    items = []
    for c in ("lo", "hi"):
        for q in ("lo", "hi"):
            for i in range(4):
                sc = max(1, min(5, (5 if c == "hi" else 2) + random.choice([-1, 0, 1])))
                items.append(Rating(obs_key=f"{c}{q}{i}", config={"C": c, "Q": q},
                                    input_id=f"in{i}", rep=0, judge_rep=0, value=sc, value_numeric=sc))
    r = Ratings(rubric=ANSWER_QUALITY_1_5, judge_model="fake", factors=["C", "Q"], items=items)
    eff = fit_effects(r)                                   # must not raise
    assert "C" in {t["factor"] for t in eff.terms}        # real factor name preserved


def test_ratings_reject_reserved_factor_names():
    """Ratings built directly (library mode, loaded bundles) get the same check as Factor:
    with factors=['error'] the stats would group on the all-None error column instead."""
    for name in ("error", "rep", "verdict", "tokens"):
        with pytest.raises(ValueError, match="reserved"):
            Ratings(rubric=ANSWER_QUALITY_1_5, judge_model="fake", factors=[name], items=[])


def test_reserved_factor_names_cover_every_bookkeeping_column():
    """Every column the tidy views write next to the factor columns must be reserved — so a
    new bookkeeping column (as energy_wh was) can't silently start colliding with a factor."""
    from cafe.evaluation import Evaluation
    from cafe.execution.results import Observation, Results
    from cafe.study import RESERVED_FACTOR_NAMES

    obs = Observation(config={}, input_id="q0", rep=0, output="x", elapsed_s=1.0)
    rating = Rating(obs_key=obs.key(), config={}, input_id="q0", rep=0, judge_rep=0,
                    value=4, value_numeric=4)
    ratings = Ratings(rubric=ANSWER_QUALITY_1_5, judge_model="fake", factors=[], items=[rating])
    answers = Results(study_name="t", factors=[], observations=[obs])
    columns = set(ratings.to_records()[0])
    columns |= set(Evaluation(study_name="t", answers=answers, ratings=ratings).records()[0])
    columns |= set(Evaluation(study_name="t", answers=answers).records()[0])  # answers-only rows
    assert columns <= RESERVED_FACTOR_NAMES, columns - RESERVED_FACTOR_NAMES


def test_fit_logistic_computes_odds_ratio():
    from cafe import fit_logistic

    # good passes 6/8, bad passes 2/8 → marginal odds ratio (good vs bad) = 3 / (1/3) = 9.
    # Force the self-contained statsmodels backend so this is deterministic without R.
    r = _binary_ratings(good_vals=[1, 1, 1, 0, 1, 1, 0, 1], bad_vals=[0, 1, 0, 0, 1, 0, 0, 0])
    log = fit_logistic(r, backend="statsmodels")
    assert log.available and len(log.terms) == 1
    term = log.terms[0]
    assert term["label"] == "method=good"
    assert term["odds_ratio"] == pytest.approx(9.0, rel=0.01)
    assert term["p"] is not None and term["p"] < 0.05
    assert log.significant_factors == ["method"]


def test_fit_logistic_glmer_backend_when_available():
    import random

    from cafe import fit_logistic
    from cafe.stats.logistic import check_glmer

    ok, _ = check_glmer()
    if not ok:
        pytest.skip("R + lme4 not installed")
    # 40 questions, good ~75% / bad ~30% pass — enough clusters for a stable GLMM.
    random.seed(1)
    good = [1 if random.random() < 0.75 else 0 for _ in range(40)]
    bad = [1 if random.random() < 0.30 else 0 for _ in range(40)]
    log = fit_logistic(_binary_ratings(good, bad), backend="glmer")
    assert log.available and "glmer" in log.model.lower()
    term = log.terms[0]
    assert term["odds_ratio"] is not None and term["odds_ratio"] > 1  # good passes more
    assert term["p"] is not None


_FIVE_FACTORS = ["model", "prompt", "retrieval", "reranker", "chunking"]
_FIVE_FACTOR_FORMULA = "verdict ~ (1 | input_id) + model + prompt + retrieval + reranker + chunking"


def _five_factor_ratings() -> Ratings:
    """Five two-level factors: the main-effects formula is longer than R deparse()'s
    60-character line width."""
    import itertools
    import random

    rng = random.Random(0)
    items = []
    for q in range(6):
        hard = rng.gauss(0, 0.5)
        for cfg in itertools.product(("a", "b"), repeat=len(_FIVE_FACTORS)):
            v = max(1, min(5, round(3 + (1.0 if cfg[0] == "b" else -0.5) + hard + rng.gauss(0, 1))))
            items.append(Rating(obs_key=f"{q}{''.join(cfg)}", config=dict(zip(_FIVE_FACTORS, cfg)),
                                input_id=f"q{q}", rep=0, judge_rep=0, value=v, value_numeric=v))
    return Ratings(rubric=ANSWER_QUALITY_1_5, judge_model="x", factors=_FIVE_FACTORS, items=items)


def test_clmm_long_formula_is_one_string():
    from cafe.stats.ordinal import check_r

    if not check_r()[0]:
        pytest.skip("R + ordinal not installed")
    res = fit_clmm(_five_factor_ratings(), interactions=1)
    assert res.available, res.reason
    assert res.formula == _FIVE_FACTOR_FORMULA          # not a list of deparse() lines
    assert f"ordinal CLMM — {_FIVE_FACTOR_FORMULA}   (n=" in res.show()


def test_glmm_script_emits_long_formula_as_one_string(tmp_path):
    import json
    import subprocess
    from importlib.resources import files

    from cafe.stats._frame import analysis_frame
    from cafe.stats.logistic import check_glmer

    if not check_glmer()[0]:
        pytest.skip("R + lme4 not installed")
    df = analysis_frame(_five_factor_ratings())
    df["verdict"] = (df["verdict"] >= 3).astype(int)
    csv = tmp_path / "d.csv"
    df[["verdict", "input_id", *_FIVE_FACTORS]].to_csv(csv, index=False)
    proc = subprocess.run(["Rscript", str(files("cafe.stats").joinpath("glmm.R")), str(csv),
                           ",".join(_FIVE_FACTORS), "1"], capture_output=True, text=True, timeout=120)
    payload = json.loads(proc.stdout)
    assert payload["available"], payload
    assert payload["formula"] == _FIVE_FACTOR_FORMULA


def test_fit_logistic_rejects_non_binary_rubric():
    from cafe import fit_logistic

    log = fit_logistic(_ratings({"good": 5, "bad": 2}))  # ordinal rubric
    assert not log.available and "binary" in log.reason
    assert "unavailable" in log.show()


def _canned_glmer(monkeypatch, optimizer, estimate=0.4):
    """fit_logistic(backend="glmer") on a canned glmm.R payload reporting ``optimizer``."""
    import json
    import subprocess

    from cafe import fit_logistic
    from cafe.stats import logistic

    payload = {"available": True, "n_obs": 16, "logLik": -9.0, "optimizer": optimizer,
               "formula": "verdict ~ (1 | input_id) + method",
               "coefficients": [{"term": "methodgood", "estimate": estimate, "std_error": 0.0014,
                                 "z": 278.3, "p": 0.0}]}
    monkeypatch.setattr(logistic, "check_glmer", lambda: (True, "ok"))
    monkeypatch.setattr(logistic.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, stdout=json.dumps(payload), stderr=""))
    return fit_logistic(_binary_ratings([1, 1, 1, 0, 1, 1, 0, 1], [0, 1, 0, 0, 1, 0, 0, 0]),
                        backend="glmer")


def test_glmer_withholds_pvalues_when_lme4_reports_no_convergence(monkeypatch):
    """glmm.R silences glmer's warnings, so lme4's "Model failed to converge" verdict must come
    through the payload; such a fit's SEs can be ~100x too small (p ≈ 0)."""
    grad = ("Model failed to converge with max|grad| = 0.0492286 (tol = 0.002, component 1)\n"
            "  See ?lme4::convergence and ?lme4::troubleshooting.")
    log = _canned_glmer(monkeypatch, {"convergence": 0, "message": "ok", "lme4_code": [-1],
                                      "lme4_messages": [grad]})
    assert log.available and log.converged is False
    t = log.terms[0]
    assert t["p"] is None and t["significant"] is False and log.significant_factors == []
    assert t["coef"] == pytest.approx(0.4) and t["odds_ratio"] is not None
    note = next(w for w in log.warnings if "did not converge" in w)
    assert "max|grad| = 0.0492286" in note and "See ?lme4" not in note
    text = log.show()
    assert "did not converge" in text and "near-degenerate" not in text and "Signif." not in text

    # a boundary (singular) fit or a clean one is converged: p-values stay
    for msgs in (["boundary (singular) fit: see help('isSingular')"], []):
        ok = _canned_glmer(monkeypatch, {"convergence": 0, "message": "ok", "lme4_code": [],
                                         "lme4_messages": msgs})
        assert ok.converged is True and ok.terms[0]["p"] == 0.0 and not ok.warnings

    # separated and not converged: both notes, the separation one still names the cause
    sep = _canned_glmer(monkeypatch, {"convergence": 0, "message": "ok", "lme4_code": [-4],
                                      "lme4_messages": ["unable to evaluate scaled gradient"]},
                        estimate=-79.3)
    assert sep.converged is False and "separates pass from fail" in sep.show()


def test_glmer_reports_a_bobyqa_refit_that_converged(monkeypatch):
    """glmm.R refits once with bobyqa when the default optimizer's fit fails lme4's checks;
    when that refit converges its SEs are valid, so p-values stay and a note says so."""
    log = _canned_glmer(monkeypatch, {"convergence": 0, "message": "ok", "lme4_code": [],
                                      "lme4_messages": [], "refit": "bobyqa"})
    assert log.converged is True and log.terms[0]["p"] == 0.0
    assert any("refit with bobyqa" in w for w in log.warnings)


def test_glmer_parses_per_factor_likelihood_ratio_tests(monkeypatch):
    from cafe import fit_logistic

    log = _canned_glmer(monkeypatch, {"convergence": 0, "message": "ok", "lme4_code": [],
                                      "lme4_messages": []})
    assert log.factor_tests == []                  # none reported -> coefficient-based fallback
    import json
    import subprocess

    from cafe.stats import logistic

    payload = {"available": True, "n_obs": 40, "logLik": -20.0, "formula": "verdict ~ (1 | input_id) + method",
               "coefficients": [{"term": "methodgood", "estimate": 79.3, "std_error": 1e4, "z": 0.0, "p": 1.0}],
               "optimizer": {"convergence": 0, "message": "ok", "lme4_code": [], "lme4_messages": []},
               "factor_tests": [{"term": "method", "factors": ["method"], "df": 1, "chisq": 40.1,
                                 "p": 2.4e-10, "converged": True}]}
    monkeypatch.setattr(logistic, "check_glmer", lambda: (True, "ok"))
    monkeypatch.setattr(logistic.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, stdout=json.dumps(payload), stderr=""))
    sep = fit_logistic(_binary_ratings(good_vals=[1] * 8, bad_vals=[0] * 8), backend="glmer")
    # separation: the Wald p-value is withheld, the likelihood-ratio test still answers
    assert sep.terms[0]["p"] is None and sep.significant_factors == ["method"]
    assert "per-factor tests (likelihood ratio" in sep.show()


def test_glmer_likelihood_ratio_tests_find_the_real_factor():
    from cafe import fit_logistic
    from cafe.stats.logistic import check_glmer

    if not check_glmer()[0]:
        pytest.skip("R + lme4 not installed")
    r = _prefix_named_factor_ratings(binary=True)              # top_k matters, retrieve doesn't
    log = fit_logistic(r, interactions=1, backend="glmer")
    tests = {t["term"]: t for t in log.factor_tests}
    assert set(tests) == {"retrieve", "retrieve.top_k"}
    assert tests["retrieve.top_k"]["significant"] and not tests["retrieve"]["significant"]
    assert fit_logistic(r, interactions=1, backend="glmer", lr_tests=False).factor_tests == []


def test_glmer_no_spurious_significance_from_unconverged_fit():
    """30 questions; B has no effect. On this data lme4 reports "Model failed to converge
    with max|grad| = 0.049" (lme4 2.0) and SE 0.0014 instead of ~0.5, so p ≈ 0. Whatever
    lme4 does on another version, B must not come out significant."""
    from cafe import fit_logistic
    from cafe.rubrics import CORRECT_PASS_FAIL
    from cafe.stats.logistic import check_glmer

    if not check_glmer()[0]:
        pytest.skip("R + lme4 not installed")
    passes = {("a1", "b1"): "110000111100101010111011100101",
              ("a2", "b1"): "110001111111001000111011000111",
              ("a1", "b2"): "111001101010001010100011100111",
              ("a2", "b2"): "110000111111101010100111100101"}
    items = [Rating(obs_key=f"{a}{b}{q}", config={"A": a, "B": b}, input_id=f"q{q + 1}", rep=0,
                    judge_rep=0, value=int(v), value_numeric=int(v))
             for (a, b), row in passes.items() for q, v in enumerate(row)]
    r = Ratings(rubric=CORRECT_PASS_FAIL, judge_model="x", factors=["A", "B"], items=items)
    log = fit_logistic(r, interactions=1, backend="glmer")
    assert log.available and "B" not in log.significant_factors
    assert log.converged is not None                      # glmm.R reports lme4's verdict
    if any("refit with bobyqa" in w for w in log.warnings):  # the bobyqa refit converged
        assert log.converged is True and all(t["p"] is not None for t in log.terms)
    if log.converged is False:
        assert all(t["p"] is None for t in log.terms)
        assert any("did not converge" in w for w in log.warnings)


def test_fit_logistic_suppresses_pvalues_under_separation():
    from cafe import fit_logistic

    # good always passes, bad always fails → perfect separation
    log = fit_logistic(_binary_ratings(good_vals=[1] * 8, bad_vals=[0] * 8), backend="statsmodels")
    assert log.available
    assert all(t["p"] is None for t in log.terms)          # not a spurious p≈0
    assert "near-degenerate" in log.show() or "separation" in log.show()


def _colon_level_ratings(binary: bool, model: str = "model", prompt: str = "prompt") -> Ratings:
    """An ollama-style level with a ':' (``llama3:8b``, the better model) plus a second
    factor, over 30 questions with a per-question difficulty."""
    import random

    from cafe.rubrics import CORRECT_PASS_FAIL

    rng = random.Random(3)
    items = []
    for q in range(30):
        hard = rng.gauss(0, 0.7)
        for m in ("gemma", "llama3:8b"):
            for p in ("plain", "cot"):
                eta = (1.5 if m == "llama3:8b" else -1.0) + hard + rng.gauss(0, 1)
                v = int(eta > 0) if binary else max(1, min(5, round(3 + eta)))
                items.append(Rating(obs_key=f"{q}{m}{p}", config={model: m, prompt: p},
                                    input_id=f"q{q}", rep=0, judge_rep=0, value=v, value_numeric=v))
    return Ratings(rubric=CORRECT_PASS_FAIL if binary else ANSWER_QUALITY_1_5, judge_model="x",
                   factors=[model, prompt], items=items)


def test_statsmodels_term_keeps_colon_inside_level():
    from cafe.stats.logistic import _parse_statsmodels_term

    back = {"cafe_f0": "model", "cafe_f1": "prompt"}
    assert _parse_statsmodels_term("C(Q('cafe_f0'))[T.llama3:8b]", back) == (
        "model=llama3:8b", "model", False)
    assert _parse_statsmodels_term("C(Q('cafe_f0'))[T.x]:y]:C(Q('cafe_f1'))[T.cot]", back) == (
        "model=x]:y × prompt=cot", "model", True)
    assert _parse_statsmodels_term("C(Q('cafe_f0'))[T.a]:C(Q('cafe_f1'))[T.b]", back) == (
        "model=a × prompt=b", "model", True)


def test_logistic_statsmodels_colon_level_is_a_main_effect():
    from cafe import fit_logistic

    log = fit_logistic(_colon_level_ratings(binary=True), interactions=2, backend="statsmodels")
    by_label = {t["label"]: t for t in log.terms}
    main = by_label["model=llama3:8b"]
    assert main["factor"] == "model" and main["interaction"] is False
    assert by_label["model=llama3:8b × prompt=plain"]["interaction"] is True
    assert log.significant_factors == ["model"]


def test_r_coef_term_prefers_the_factors_and_levels_r_reports():
    from cafe.stats.ordinal import _r_coef_term

    fs = ["model", "prompt"]
    c = {"term": "modelllama3:8b:promptcot", "factors": ["model", "prompt"],
         "levels": ["llama3:8b", "cot"]}
    assert _r_coef_term(c, fs) == ("model=llama3:8b × prompt=cot", "model", True)
    c = {"term": "`top-k`5", "factors": ["top-k"], "levels": ["5"]}
    assert _r_coef_term(c, ["top-k"]) == ("top-k=5", "top-k", False)
    # no structured info -> R's name is parsed as before
    assert _r_coef_term({"term": "modelgpt:promptcot"}, fs) == ("model=gpt × prompt=cot", "model", True)


@pytest.mark.parametrize("model", ["model", "llm model"])   # plain and non-syntactic (back-quoted)
def test_clmm_colon_level_is_a_main_effect(model):
    from cafe.stats.ordinal import check_r

    if not check_r()[0]:
        pytest.skip("R + ordinal not installed")
    clmm = fit_clmm(_colon_level_ratings(binary=False, model=model), interactions=2)
    assert clmm.available, clmm.reason
    by_label = {c["label"]: c for c in clmm.coefficients}
    main = by_label[f"{model}=llama3:8b"]
    assert main["factor"] == model and main["interaction"] is False
    assert by_label[f"{model}=llama3:8b × prompt=plain"]["interaction"] is True
    assert clmm.significant_factors == [model]


@pytest.mark.parametrize("model", ["model", "llm model"])
def test_glmer_colon_level_is_a_main_effect(model):
    from cafe import fit_logistic
    from cafe.stats.logistic import check_glmer

    if not check_glmer()[0]:
        pytest.skip("R + lme4 not installed")
    log = fit_logistic(_colon_level_ratings(binary=True, model=model), interactions=2,
                       backend="glmer")
    assert log.available and "glmer" in log.model
    by_label = {t["label"]: t for t in log.terms}
    main = by_label[f"{model}=llama3:8b"]
    assert main["factor"] == model and main["interaction"] is False
    assert by_label[f"{model}=llama3:8b × prompt=plain"]["interaction"] is True
    assert log.significant_factors == [model]


def test_report_routes_model_by_scale_type():
    from cafe.evaluation import Evaluation
    from cafe.execution.results import Results

    def report_for(ratings):
        ev = Evaluation(study_name="t",
                        answers=Results(study_name="t", factors=["method"]),
                        ratings=ratings, attribution=attribute(ratings))
        return ev.report()

    binary = report_for(_binary_ratings([1, 1, 1, 0, 1, 1, 0, 1], [0, 1, 0, 0, 1, 0, 0, 0]))
    assert "LOGISTIC" in binary and "ORDINAL" not in binary and "INFERENTIAL" not in binary

    ordinal = report_for(_ratings({"good": 5, "bad": 2}))
    assert "INFERENTIAL" in ordinal and "ORDINAL" in ordinal and "LOGISTIC" not in ordinal


def test_judge_stability_measures_per_answer_spread():
    from cafe import judge_stability

    items = []
    verdicts = {"q0": [5, 5, 5], "q1": [2, 4, 4], "q2": [5, 5, 1]}
    for iid, vs in verdicts.items():
        for jr, v in enumerate(vs):
            items.append(Rating(obs_key=f"{iid}::0", config={}, input_id=iid, rep=0,
                                judge_rep=jr, value=v, value_numeric=v))
    r = Ratings(rubric=ANSWER_QUALITY_1_5, judge_model="J", factors=[], items=items)

    st = judge_stability(r)
    assert st.judge_model == "J" and st.n_answers == 3 and st.judge_reps == 3
    by_id = {row["input_id"]: row for row in st.per_answer}
    assert by_id["q0"]["sd"] == 0.0
    assert by_id["q2"]["sd"] == pytest.approx(1.8856, abs=1e-3)   # pstdev([5,5,1])
    assert by_id["q2"]["range"] == 4
    assert st.unanimous_frac == pytest.approx(1 / 3)              # only q0 is unanimous
    assert st.max_sd == pytest.approx(1.8856, abs=1e-3)


def test_judge_stability_empty_without_repetitions():
    from cafe import judge_stability

    items = [Rating(obs_key="a::0", config={}, input_id="q0", rep=0, judge_rep=0,
                    value=4, value_numeric=4)]
    st = judge_stability(Ratings(rubric=ANSWER_QUALITY_1_5, judge_model="J", factors=[], items=items))
    assert st.n_answers == 0 and st.mean_sd is None
    assert "judge_replications" in st.show()   # explains why it's empty


def test_judge_reps_averaged_system_reps_kept():
    from cafe.stats._frame import analysis_frame

    items = []
    # one answer (q0, rep 0) judged 3x -> averaged to a single verdict
    for jr, s in enumerate([5, 5, 3]):
        items.append(Rating(obs_key="a-0-0", config={"method": "a"}, input_id="q0",
                            rep=0, judge_rep=jr, value=s, value_numeric=s))
    # a SECOND system rep of the same question (q0, rep 1) -> kept as its own row
    items.append(Rating(obs_key="a-0-1", config={"method": "a"}, input_id="q0",
                        rep=1, judge_rep=0, value=1, value_numeric=1))
    r = Ratings(rubric=ANSWER_QUALITY_1_5, judge_model="fake", factors=["method"], items=items)

    df = analysis_frame(r)
    assert len(df) == 2                              # 2 answers (system reps), not 4 ratings
    assert sorted(df["verdict"].tolist()) == [1, 4]  # 4 = round(mean(5,5,3)); rep 1 stays 1

    attr = attribute(r)
    assert attr.n_ratings == 4   # raw ratings recorded
    assert attr.n_usable == 2    # answers actually entering the stats


def _question_effect_ratings(binary: bool, question_sd: float, seed: int) -> Ratings:
    """model (large is better) × prompt (no effect) over 30 questions whose difficulty has
    standard deviation ``question_sd`` on the latent logit scale (0 = no question effect)."""
    import random

    from cafe.rubrics import CORRECT_PASS_FAIL

    rng = random.Random(seed)
    items = []
    for q in range(30):
        u = rng.gauss(0, question_sd) if question_sd else 0.0
        for m in ("small", "large"):
            for p in ("plain", "cot"):
                eta = (1.0 if m == "large" else -0.5) + u + rng.gauss(0, 1)
                v = int(eta > 0) if binary else max(1, min(5, round(3 + eta)))
                items.append(Rating(obs_key=f"{q}{m}{p}", config={"model": m, "prompt": p},
                                    input_id=f"q{q}", rep=0, judge_rep=0, value=v, value_numeric=v))
    return Ratings(rubric=CORRECT_PASS_FAIL if binary else ANSWER_QUALITY_1_5, judge_model="x",
                   factors=["model", "prompt"], items=items)


def _assert_latent_variance_components(vc):
    import math

    assert set(vc) == {"random_intercept", "random_intercept_sd", "residual", "icc"}
    assert vc["random_intercept_sd"] == pytest.approx(math.sqrt(vc["random_intercept"]), rel=1e-4)
    assert vc["residual"] == pytest.approx(math.pi ** 2 / 3)
    assert vc["icc"] == pytest.approx(vc["random_intercept"] / (vc["random_intercept"] + math.pi ** 2 / 3))


def test_clmm_reports_random_intercept_variance():
    from cafe.stats.ordinal import check_r

    if not check_r()[0]:
        pytest.skip("R + ordinal not installed")
    res = fit_clmm(_question_effect_ratings(binary=False, question_sd=1.5, seed=5), interactions=1)
    assert res.available, res.reason
    _assert_latent_variance_components(res.variance_components)
    assert res.variance_components["random_intercept"] > 1      # a real between-question effect
    assert "between-question σ_u²" in res.show() and "ICC" in res.show()
    assert not any("near-singular" in w for w in res.warnings)

    flat = fit_clmm(_question_effect_ratings(binary=False, question_sd=0.0, seed=7), interactions=1)
    assert flat.variance_components["random_intercept_sd"] < 1e-3   # collapsed
    assert any("near-singular" in w for w in flat.warnings)


def test_glmer_reports_random_intercept_variance():
    from cafe import fit_logistic
    from cafe.stats.logistic import check_glmer

    if not check_glmer()[0]:
        pytest.skip("R + lme4 not installed")
    log = fit_logistic(_question_effect_ratings(binary=True, question_sd=1.5, seed=5),
                       interactions=1, backend="glmer")
    assert log.available and "glmer" in log.model
    _assert_latent_variance_components(log.variance_components)
    assert log.variance_components["random_intercept"] > 1
    assert "between-question σ_u²" in log.show() and "ICC" in log.show()
    assert not any("near-singular" in w for w in log.warnings)

    flat = fit_logistic(_question_effect_ratings(binary=True, question_sd=0.0, seed=7),
                        interactions=1, backend="glmer")
    assert flat.variance_components["random_intercept"] == pytest.approx(0, abs=1e-6)
    assert any("near-singular" in w for w in flat.warnings)


def test_random_intercept_components_and_note_from_r_payload():
    import math

    from cafe.stats.logistic import Logistic
    from cafe.stats.ordinal import CLMMResult, _set_random_intercept

    res = CLMMResult(available=True, formula="f", n_obs=10)
    _set_random_intercept(res, {"random_intercept": {"variance": 0.25, "sd": 0.5}})
    assert res.variance_components["icc"] == pytest.approx(0.25 / (0.25 + math.pi ** 2 / 3))
    assert not res.warnings
    assert "between-question σ_u² = 0.250 (σ_u = 0.500)" in res.show()

    log = Logistic(available=True, formula="f", n_obs=10)
    _set_random_intercept(log, {"random_intercept": {"variance": 0.0, "sd": 0.0}})
    assert log.variance_components["icc"] == 0.0
    assert any("near-singular" in w for w in log.warnings)
    assert "ICC = 0.000" in log.show() and "near-singular" in log.show()

    statsmodels_like = Logistic(available=True, formula="f")
    _set_random_intercept(statsmodels_like, {})                 # no random intercept fit
    assert statsmodels_like.variance_components is None
