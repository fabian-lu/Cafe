"""Inspecting a finished evaluation: failures() + the unified records() view."""

import cafe
from cafe.evaluation import Evaluation
from cafe.execution.results import Observation, Results
from cafe.judging.ratings import Rating, Ratings


def _evaluation():
    cfg = {"model": "m"}
    o0 = Observation(config=cfg, input_id="q0", rep=0, output="Paris",
                     elapsed_s=1.2, metadata={"cost_usd": 0.01, "tokens": 50})
    o1 = Observation(config=cfg, input_id="q1", rep=0, output="banana", elapsed_s=0.9)
    answers = Results(study_name="t", factors=["model"], observations=[o0, o1])
    ratings = Ratings(
        rubric=cafe.ANSWER_QUALITY_1_5, judge_model="j", factors=["model"],
        judge_system_prompt="You are strict.",
        items=[
            Rating(obs_key=o0.key(), config=cfg, input_id="q0", rep=0, judge_rep=0,
                   value=5, value_numeric=5, reasoning="correct",
                   prompt="grade this", raw_response="GRADE: 5"),
            Rating(obs_key=o1.key(), config=cfg, input_id="q1", rep=0, judge_rep=0,
                   error="no GRADE marker", prompt="grade this",
                   raw_response="i think it's fine"),
        ],
    )
    return Evaluation(study_name="t", answers=answers, ratings=ratings,
                      questions={"q0": "Capital of France?", "q1": "What is 2+2?"},
                      references={"q0": "Paris", "q1": "4"})


def test_failures_lists_unparseable_with_prompt_and_raw():
    fails = _evaluation().ratings.failures()
    assert len(fails) == 1
    assert fails[0]["input_id"] == "q1"
    assert fails[0]["raw_response"] == "i think it's fine"
    assert fails[0]["error"]


def test_records_join_everything():
    r0 = next(r for r in _evaluation().records() if r["input_id"] == "q0")
    assert r0["question"] == "Capital of France?"
    assert r0["reference"] == "Paris"
    assert r0["answer"] == "Paris"
    assert r0["model"] == "m"          # factor flattened into the row
    assert r0["verdict"] == 5
    assert (r0["cost_usd"], r0["tokens"]) == (0.01, 50)
    assert r0["judge_system"] == "You are strict."   # full judge input reconstructable
    assert r0["judge_raw"] == "GRADE: 5"


def test_report_surfaces_failed_answers():
    cfg = {"model": "m"}
    ok = Observation(config=cfg, input_id="q0", rep=0, output="ans")
    bad = Observation(config=cfg, input_id="q1", rep=0, output=None, error="LLMError: timeout")
    answers = Results(study_name="t", factors=["model"], observations=[ok, bad])
    ratings = Ratings(rubric=cafe.ANSWER_QUALITY_1_5, judge_model="j", factors=["model"],
                      items=[Rating(obs_key=ok.key(), config=cfg, input_id="q0", rep=0,
                                    judge_rep=0, value=5, value_numeric=5)])
    ev = Evaluation(study_name="t", answers=answers, ratings=ratings,
                    attribution=cafe.attribute(ratings))
    rep = ev.report()
    assert "pipeline: 2 answers (1 failed to generate)" in rep
    assert "failed to generate" in rep and "result.answers.errors" in rep
    assert "(1 failed)" in ev.show()


def test_records_without_judge_falls_back_to_answers():
    ev = _evaluation()
    ev.ratings = None
    rows = ev.records()
    assert {r["answer"] for r in rows} == {"Paris", "banana"}
    assert all("verdict" not in r for r in rows)
