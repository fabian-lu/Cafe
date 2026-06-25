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
