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


def test_interactions_detected_and_toggleable():
    r = _two_factor_interaction_ratings()
    two_way = {t["factor"]: t for t in fit_effects(r, interactions=2).terms}
    assert "A × B" in two_way and two_way["A × B"]["interaction"] and two_way["A × B"]["significant"]
    main_only = {t["factor"] for t in fit_effects(r, interactions=1).terms}
    assert "A × B" not in main_only and {"A", "B"} <= main_only


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_clmm_accepts_interaction_order():
    res = fit_clmm(_two_factor_interaction_ratings(), interactions=2)
    # Either a fit (possibly with an interaction coefficient) or an honest reason.
    assert res.available or res.reason


def test_readable_clmm_term():
    from cafe.stats.ordinal import _readable_term

    fs = ["retrieve", "retrieve.top_k", "rerank"]
    assert _readable_term("retrievekeyword", fs) == "retrieve=keyword"
    assert _readable_term("retrieve.top_k2", fs) == "retrieve.top_k=2"   # longest-prefix wins
    assert _readable_term("retrievekeyword:reranknone", fs) == "retrieve=keyword × rerank=none"


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


def test_fit_logistic_rejects_non_binary_rubric():
    from cafe import fit_logistic

    log = fit_logistic(_ratings({"good": 5, "bad": 2}))  # ordinal rubric
    assert not log.available and "binary" in log.reason
    assert "unavailable" in log.show()


def test_fit_logistic_suppresses_pvalues_under_separation():
    from cafe import fit_logistic

    # good always passes, bad always fails → perfect separation
    log = fit_logistic(_binary_ratings(good_vals=[1] * 8, bad_vals=[0] * 8), backend="statsmodels")
    assert log.available
    assert all(t["p"] is None for t in log.terms)          # not a spurious p≈0
    assert "near-degenerate" in log.show() or "separation" in log.show()


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
