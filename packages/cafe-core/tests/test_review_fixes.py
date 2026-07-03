"""Regression tests for the fixes from the 2026-07 code review (review/FINAL_FINDINGS.md)."""

import warnings

import pytest

import cafe
from cafe.execution.results import Observation, Results
from cafe.judging.ratings import JudgeOutput, Rating, Ratings
from cafe.judging.runner import judge_results


class _FixedJudge:
    model = "fixed"

    async def score(self, rubric, question, answer, reference=None):
        # empty answer → 1, else 5 (so we can tell empty answers were actually judged)
        v = 1 if answer == "" else 5
        return JudgeOutput(v, v, "ok", "prompt", "raw")


def _results(outputs):
    obs = [Observation(config={"m": "a"}, input_id=f"q{i}", rep=0, output=o)
           for i, o in enumerate(outputs)]
    return Results(study_name="t", factors=["m"], observations=obs)


# ── C-1: empty-string answers are judged; None answers become explicit error rows ──
async def test_empty_string_answer_is_judged_none_is_error():
    results = _results(["real answer", "", None])
    ratings = await judge_results(results, _FixedJudge(), cafe.ANSWER_QUALITY_1_5)
    scored = {r.input_id: r.value_numeric for r in ratings.items}
    assert scored["q0"] == 5           # real answer judged
    assert scored["q1"] == 1           # EMPTY string judged (not dropped)
    assert scored["q2"] is None        # None → unjudgeable error row
    none_row = next(r for r in ratings.items if r.input_id == "q2")
    assert "unjudgeable" in (none_row.error or "")


# ── C-2: normalize_output warns on a mis-shaped envelope, not on a genuine dict answer ──
def test_normalize_output_warns_on_stray_metadata():
    from cafe.system import normalize_output

    with pytest.warns(UserWarning, match="without an 'output' key"):
        out, meta = normalize_output({"cost_usd": 0.01, "latency_s": 0.5})
    assert meta == {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a genuine JSON answer must NOT warn
        normalize_output({"answer": "hi", "score": 5})


# ── C-5: a tied binary judge split rounds up (pass), not down (banker's rounding) ──
def test_binary_tie_rounds_half_up():
    from cafe.stats._frame import analysis_frame

    items = [Rating(obs_key="a", config={"m": "x"}, input_id="q0", rep=0, judge_rep=j,
                    value=v, value_numeric=v) for j, v in enumerate([1, 0])]
    r = Ratings(rubric=cafe.rubrics.CORRECT_PASS_FAIL, judge_model="f", factors=["m"], items=items)
    assert analysis_frame(r)["verdict"].tolist() == [1]  # 0.5 → 1, not 0


# ── C-3: resuming an edited study drops stale checkpoint rows ──
async def test_resume_drops_stale_checkpoint_rows(tmp_path):
    from cafe.execution import run_study

    ckpt = str(tmp_path / "run.jsonl")
    s1 = cafe.Study(name="s", system=lambda c, i: f"{c['x']}-{i['text']}",
                    factors=[cafe.Factor("x", ["a", "b"])],
                    dataset=[{"id": "q0", "text": "q0"}])
    await run_study(s1, checkpoint_path=ckpt, progress=False)
    # change the design: x levels become c/d
    s2 = cafe.Study(name="s", system=lambda c, i: f"{c['x']}-{i['text']}",
                    factors=[cafe.Factor("x", ["c", "d"])],
                    dataset=[{"id": "q0", "text": "q0"}])
    with pytest.warns(UserWarning, match="no longer in the study's design"):
        res = await run_study(s2, checkpoint_path=ckpt, progress=False)
    levels = {o.config["x"] for o in res.observations}
    assert levels == {"c", "d"}  # the old a/b ghost rows are gone


# ── C-7: judging is checkpointed and resumes ──
async def test_judging_checkpoint_resumes(tmp_path):
    ckpt = str(tmp_path / "judge.jsonl")
    results = _results(["one", "two", "three"])
    r1 = await judge_results(results, _FixedJudge(), cafe.ANSWER_QUALITY_1_5, checkpoint_path=ckpt)
    assert len(r1.items) == 3
    # a fresh judge that would score differently — but everything is already checkpointed
    class _Other(_FixedJudge):
        model = "other"

        async def score(self, rubric, question, answer, reference=None):
            return JudgeOutput(2, 2, "different", "p", "r")

    r2 = await judge_results(results, _Other(), cafe.ANSWER_QUALITY_1_5, checkpoint_path=ckpt)
    assert [r.value_numeric for r in r2.items] == [5, 5, 5]  # resumed, not re-scored


# ── C-10: Study.check() guardrails ──
def test_check_warns_single_factor_and_judge_rubric_mismatch():
    data = [{"text": f"q{i}"} for i in range(10)]
    single = cafe.Study(name="t", system=lambda c, i: "x",
                        factors=[cafe.Factor("a", ["x", "y"])], dataset=data,
                        rubric=cafe.ANSWER_QUALITY_1_5, judge=cafe.LLMJudge(model="m"))
    assert any("only one factor" in w for w in single.check())
    mismatch = cafe.Study(name="t", system=lambda c, i: "x",
                          factors=[cafe.Factor("a", ["x", "y"]), cafe.Factor("b", ["p", "q"])],
                          dataset=data, judge=cafe.LLMJudge(model="m"))  # judge, no rubric
    assert any("no rubric" in w for w in mismatch.check())


def test_check_warns_reps_at_temp_zero_and_continuous():
    data = [{"text": f"q{i}"} for i in range(10)]
    s = cafe.Study(name="t", system=lambda c, i: "x",
                   factors=[cafe.Factor("k", [1, 2, 4], type=cafe.FactorType.continuous),
                            cafe.Factor("b", ["p", "q"])],
                   dataset=data, rubric=cafe.ANSWER_QUALITY_1_5,
                   judge=cafe.LLMJudge(model="m", temperature=0.0), judge_replications=3)
    warns = s.check()
    assert any("temperature is 0.0" in w for w in warns)
    assert any("declared continuous" in w for w in warns)


# ── C-16: R-style accessors ──
def test_rstyle_accessors_present():
    from cafe.stats import attribute
    from cafe.evaluation import Evaluation

    items = []
    for i in range(8):
        for m, sc in (("good", 5), ("bad", 2)):
            items.append(Rating(obs_key=f"{m}{i}", config={"method": m}, input_id=f"q{i}",
                                rep=0, judge_rep=0, value=sc, value_numeric=sc))
    r = Ratings(rubric=cafe.ANSWER_QUALITY_1_5, judge_model="f", factors=["method"], items=items)
    ev = Evaluation(study_name="t", answers=Results(study_name="t", factors=["method"]),
                    ratings=r, attribution=attribute(r))
    assert list(ev.effects.to_df().columns)[:2] == ["factor", "interaction"]
    assert set(ev.marginal_means["level"]) == {"good", "bad"}
    assert ev.residuals is not None and len(ev.residuals) == 16
