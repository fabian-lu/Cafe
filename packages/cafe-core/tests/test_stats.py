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
