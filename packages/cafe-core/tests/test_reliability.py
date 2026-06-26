"""Inter-rater reliability (Krippendorff's alpha) + human ratings ingestion."""

import math

import pytest

import cafe
from cafe.evaluation import Evaluation
from cafe.execution.results import Observation, Results
from cafe.judging.ratings import Rating, Ratings
from cafe.stats.reliability import krippendorff_alpha


def test_alpha_perfect_agreement():
    table = {"a": {1: 5, 2: 4, 3: 3}, "b": {1: 5, 2: 4, 3: 3}}
    assert krippendorff_alpha(table) == 1.0


def test_alpha_handles_missing_data():
    table = {"a": {1: 5, 2: 4, 3: 3, 4: 2}, "b": {1: 5, 2: 4, 3: 3}}  # b missing unit 4
    assert krippendorff_alpha(table) == 1.0  # agree on every jointly-rated unit


def test_alpha_disagreement_is_negative():
    table = {"a": {1: 1, 2: 1, 3: 1}, "b": {1: 5, 2: 5, 3: 5}}
    assert krippendorff_alpha(table, "ordinal") < 0


def test_human_ratings_validates_columns():
    with pytest.raises(ValueError, match="missing"):
        cafe.human_ratings([{"answer_id": "x", "rater": "ann"}])  # no score


def test_human_ratings_roundtrip():
    hr = cafe.human_ratings([
        {"answer_id": "x", "rater": "ann", "score": 5},
        {"answer_id": "x", "rater": "bob", "score": 4},
    ])
    assert hr.raters() == ["ann", "bob"]
    assert hr.by_rater()["ann"]["x"] == 5


def _evaluation_with_judge():
    config = {"sys": "a"}
    obs, ratings = [], []
    judge_scores = {0: 5, 1: 4, 2: 5, 3: 3}
    for i, sc in judge_scores.items():
        o = Observation(config=config, input_id=f"q{i}", rep=0, output="ans")
        obs.append(o)
        ratings.append(Rating(obs_key=o.key(), config=config, input_id=f"q{i}",
                              rep=0, judge_rep=0, value=sc, value_numeric=sc))
    answers = Results(study_name="t", factors=["sys"], observations=obs)
    rt = Ratings(rubric=cafe.ANSWER_QUALITY_1_5, judge_model="m", factors=["sys"], items=ratings)
    return Evaluation(study_name="t", answers=answers, ratings=rt), [o.key() for o in obs]


def test_reliability_judge_vs_human():
    ev, keys = _evaluation_with_judge()
    # human agrees with the judge exactly
    human = [{"answer_id": k, "rater": "ann", "score": s}
             for k, s in zip(keys, [5, 4, 5, 3])]
    rel = cafe.reliability(ev, human=human)
    assert set(rel.raters) == {"judge", "ann"}
    assert rel.alpha == 1.0
    assert rel.n_units == 4
    assert rel.interpret(rel.alpha) == "reliable"


def test_reliability_needs_two_raters():
    ev, _ = _evaluation_with_judge()
    with pytest.raises(ValueError, match="two raters"):
        cafe.reliability(ev)  # judge only


def test_reliability_table_path_and_pairwise():
    table = {
        "j": {1: 5, 2: 4, 3: 3, 4: 2},
        "ann": {1: 5, 2: 4, 3: 3, 4: 2},
        "bob": {1: 1, 2: 2, 3: 3, 4: 4},
    }
    rel = cafe.reliability(table=table)
    assert len(rel.pairwise) == 3
    j_ann = next(p for p in rel.pairwise if {p["a"], p["b"]} == {"j", "ann"})
    assert math.isclose(j_ann["alpha"], 1.0)


def test_answer_sheet_has_stable_ids():
    ev, keys = _evaluation_with_judge()
    sheet = cafe.answer_sheet(ev)
    assert [r["answer_id"] for r in sheet] == keys
    assert all("output" in r for r in sheet)
