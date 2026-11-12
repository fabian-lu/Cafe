"""Inspecting a finished evaluation: failures() + the unified records() view."""

import warnings

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
    assert r0["output"] == "Paris"
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
    assert {r["output"] for r in rows} == {"Paris", "banana"}
    assert all("verdict" not in r for r in rows)


def test_report_tells_failed_judge_calls_from_unparseable_verdicts():
    cfg = {"model": "m"}
    obs = [Observation(config=cfg, input_id=f"q{i}", rep=0, output="ans") for i in range(4)]
    obs.append(Observation(config=cfg, input_id="q4", rep=0, output=None))
    errors = [
        None,
        "no GRADE marker in judge output: 'it depends'",
        "judge call failed: model 'm' needs OLLAMA_API_KEY (set it in .env).",
        "judge raised ConnectionError: 503 from grading service",
        "unjudgeable: the system produced no output (None)",
    ]
    ratings = Ratings(rubric=cafe.ANSWER_QUALITY_1_5, judge_model="j", factors=["model"], items=[
        Rating(obs_key=o.key(), config=cfg, input_id=o.input_id, rep=0, judge_rep=0,
               value=None if e else 4, value_numeric=None if e else 4, error=e)
        for o, e in zip(obs, errors)])
    ev = Evaluation(study_name="t", answers=Results(study_name="t", factors=["model"],
                                                    observations=obs), ratings=ratings)

    assert ev._pipeline_line() == (
        "pipeline: 5 answers  →  5 judged (1 unparseable, 2 judge call(s) failed, "
        "1 with no output to judge)  →  1 usable verdict(s)")
    warns = ev._health_warnings()
    assert len(warns) == 3
    assert warns[0].startswith("1 answer(s) produced no output")
    assert warns[1].startswith("2 judge call(s) failed, so those answers have no verdict")
    assert warns[1].endswith("The first error: judge call failed: model 'm' needs "
                             "OLLAMA_API_KEY (set it in .env).")
    assert warns[2] == "1 verdict(s) were unparseable — inspect result.ratings.failures()"


def test_judge_calls_that_fail_are_not_called_unparseable(monkeypatch):
    # No API key: every judge call fails before any reply exists to parse.
    async def failing(model, messages, **kwargs):
        raise cafe.LLMError("model 'ollama_cloud/gpt-oss:20b' needs OLLAMA_API_KEY (set it in .env).")

    monkeypatch.setattr("cafe.judging.judge.complete", failing)
    study = cafe.Study(name="t", system=lambda c, i: "x", factors=[cafe.Factor("m", ["a", "b"])],
                       dataset=["q0", "q1", "q2", "q3"], rubric=cafe.rubrics.CORRECTNESS_0_3,
                       judge=cafe.LLMJudge("ollama_cloud/gpt-oss:20b", structured=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the design advisories
        ev = study.evaluate(progress=False)
    assert "8 judged (8 judge call(s) failed)" in ev._pipeline_line()  # was: (8 unparseable)
    [warning] = ev._health_warnings()
    assert warning.startswith("8 judge call(s) failed") and "needs OLLAMA_API_KEY" in warning
