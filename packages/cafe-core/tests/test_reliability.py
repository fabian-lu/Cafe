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
    assert all("output" in r and r["score"] == "" for r in sheet)


def test_answer_sheet_csv_roundtrip_skips_blanks(tmp_path):
    ev, keys = _evaluation_with_judge()
    path = str(tmp_path / "sheet.csv")
    rows = cafe.answer_sheet(ev, path, raters=("expert_1", "expert_2"))
    assert len(rows) == 2 * len(keys)          # one row per (answer × expert)
    import pandas as pd
    df = pd.read_csv(path)
    df.loc[df["answer_id"] == keys[0], "score"] = 5   # only one answer rated; rest blank
    df.to_csv(path, index=False)
    hr = cafe.human_ratings(path)               # reads CSV, skips the blank rows
    assert hr.raters() == ["expert_1", "expert_2"]
    assert {r["answer_id"] for r in hr.records} == {keys[0]}


def test_reliability_auto_metric_from_rubric():
    ev, keys = _evaluation_with_judge()  # ANSWER_QUALITY_1_5 is ordinal
    human = [{"answer_id": k, "rater": "ann", "score": s} for k, s in zip(keys, [5, 4, 5, 3])]
    assert cafe.reliability(ev, human=human).metric == "ordinal"


def test_rejudge_and_raters_judge_vs_judge():
    from cafe.judging.ratings import JudgeOutput

    class FakeJudge:
        model = "judge-B"

        async def score(self, rubric, question, answer, reference=None):
            v = 4 if "5" in str(answer) else 3   # deterministic, no LLM
            return JudgeOutput(v, v, "fake", "prompt", "raw")

    ev, keys = _evaluation_with_judge()
    ev_b = ev.rejudge(FakeJudge(), progress=False)     # same answers, new judge, sync
    assert ev_b.ratings.judge_model == "judge-B"
    assert [r.obs_key for r in ev_b.ratings.items] == keys   # same answers rejudged
    rel = cafe.reliability(raters={"A": ev, "B": ev_b})
    assert set(rel.raters) == {"A", "B"}


def test_custom_non_llm_judge_satisfies_protocol_and_drives_rejudge():
    from cafe.judging.ratings import JudgeOutput

    class LengthJudge:  # no LLM, purely programmatic
        model = "length-grader"

        async def score(self, rubric, question, answer, reference=None):
            v = 1 if len(answer or "") >= 3 else 0
            return JudgeOutput(v, v, "programmatic", "(no prompt)", None)

    assert isinstance(LengthJudge(), cafe.Judge)   # duck-typed protocol
    ev, keys = _evaluation_with_judge()
    graded = ev.rejudge(LengthJudge(), rubric=cafe.rubrics.CORRECT_PASS_FAIL, progress=False)
    assert graded.ratings.judge_model == "length-grader"
    assert graded.overall_mean == 1.0   # every stored answer is "ans" (len 3)


def test_rejudge_respects_rubric_and_repetitions():
    from cafe.judging.ratings import JudgeOutput

    class PassFail:
        model = "pf"

        async def score(self, rubric, question, answer, reference=None):
            return JudgeOutput(1, 1, "pass", "prompt", "raw")

    ev, keys = _evaluation_with_judge()
    # different rubric (binary) applied to the same answers
    binary = cafe.rubrics.CORRECT_PASS_FAIL
    ev_b = ev.rejudge(PassFail(), rubric=binary, progress=False)
    assert ev_b.ratings.rubric is binary
    assert {r.obs_key for r in ev_b.ratings.items} == set(keys)
    # repetitions multiply the verdicts per answer, without regenerating answers
    ev_r = ev.rejudge(PassFail(), repetitions=3, progress=False)
    assert len(ev_r.ratings.items) == 3 * len(keys)
    assert len(ev_r.answers.observations) == len(ev.answers.observations)


def test_human_evaluation_runs_full_stats():
    ev, keys = _evaluation_with_judge()
    human = [{"answer_id": k, "rater": r, "score": s}
             for k, s in zip(keys, [5, 4, 2, 1]) for r in ("e1", "e2")]
    hev = cafe.human_evaluation(ev, human)
    assert hev.ratings.judge_model == "human"
    assert hev.attribution.n_usable == len(keys)   # 2 experts collapsed per answer
    assert hev.effects is not None
