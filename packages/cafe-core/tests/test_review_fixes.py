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


# ── judging resume keeps only the current answers × repetitions (issue #11) ──
async def test_judging_resume_drops_stale_and_excess_checkpoint_rows(tmp_path):
    ckpt = str(tmp_path / "judge.jsonl")
    results = _results(["one", "two", "three"])
    r1 = await judge_results(results, _FixedJudge(), cafe.ANSWER_QUALITY_1_5,
                             repetitions=2, checkpoint_path=ckpt)
    assert len(r1.items) == 6

    # fewer repetitions than checkpointed → the judge_rep=1 rows are not part of this run
    with pytest.warns(UserWarning, match="dropped 3 stored verdict"):
        r2 = await judge_results(results, _FixedJudge(), cafe.ANSWER_QUALITY_1_5,
                                 repetitions=1, checkpoint_path=ckpt)
    assert len(r2.items) == 3
    assert {r.judge_rep for r in r2.items} == {0}

    # a different Results (the study changed) → the old config's verdicts must not leak in
    other = Results(study_name="t", factors=["m"],
                    observations=[Observation(config={"m": "new"}, input_id="q9",
                                              rep=0, output="fresh")])
    progress = []
    with pytest.warns(UserWarning, match="dropped 6 stored verdict"):
        r3 = await judge_results(other, _FixedJudge(), cafe.ANSWER_QUALITY_1_5,
                                 checkpoint_path=ckpt,
                                 on_progress=lambda _r, done, total: progress.append((done, total)))
    assert [r.config for r in r3.items] == [{"m": "new"}]
    assert progress == [(1, 1)]


async def test_judging_resume_retries_errors_and_counts_them_once(tmp_path):
    class _FlakyJudge(_FixedJudge):
        failed = False

        async def score(self, rubric, question, answer, reference=None):
            if answer == "two" and not _FlakyJudge.failed:  # fails once, then recovers
                _FlakyJudge.failed = True
                return JudgeOutput(None, None, "timeout", "prompt", None)
            return JudgeOutput(4, 4, "ok", "prompt", "raw")

    ckpt = str(tmp_path / "judge.jsonl")
    results = _results(["one", "two", "three"])
    r1 = await judge_results(results, _FlakyJudge(), cafe.ANSWER_QUALITY_1_5, checkpoint_path=ckpt)
    assert len(r1.errors) == 1

    # retry_errors=False: the failed verdict counts as done, nothing is re-judged
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a retryable error is not a stale row
        kept = await judge_results(results, _FlakyJudge(), cafe.ANSWER_QUALITY_1_5,
                                   checkpoint_path=ckpt, retry_errors=False)
    assert len(kept.items) == 3 and len(kept.errors) == 1

    progress = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r2 = await judge_results(results, _FlakyJudge(), cafe.ANSWER_QUALITY_1_5,
                                 checkpoint_path=ckpt,
                                 on_progress=lambda _r, done, total: progress.append((done, total)))
    assert len(r2.items) == 3 and not r2.errors  # the retry replaced the failed verdict
    assert progress == [(3, 3)]  # 2 carried forward + 1 retried, not 3 + 1


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


# ── NumPy levels keep their type and their answers' keys through the JSON files ──
class _LengthJudge:
    model = "length"

    async def score(self, rubric, question, answer, reference=None):
        return JudgeOutput(len(answer) % 5 + 1, len(answer) % 5 + 1, "ok", "prompt", "raw")


def test_config_id_reads_a_numpy_number_as_the_number():
    np = pytest.importorskip("numpy")
    from cafe.execution.results import config_id

    assert config_id({"k": np.int64(3)}) == config_id({"k": 3}) != config_id({"k": "3"})
    assert config_id({"k": np.bool_(True)}) == config_id({"k": True})
    assert config_id({"k": np.float32(0.5)}) == config_id({"k": 0.5})


async def test_numpy_levels_keep_their_type_through_a_saved_evaluation(tmp_path):
    """save_evaluation wrote np.int64(3) with default=str, so load_evaluation brought the level
    back as the string "3": the marginals sorted "10" before "3" and records() changed type.
    It is written as the number now, and config_id encodes it the same way, so the answers
    keep their keys: every verdict finds its answer, and an answer sheet exported before
    saving matches the reloaded answers."""
    np = pytest.importorskip("numpy")
    from cafe.evaluation import evaluate, load_evaluation, save_evaluation

    study = cafe.Study(name="np", system=lambda config, item: f"{config}:{item}",
                       factors=[cafe.Factor("top_k", [np.int64(3), np.int64(10)]),
                                cafe.Factor("m", ["a", "b"])],
                       dataset=[f"q{i}" for i in range(8)], rubric=cafe.ANSWER_QUALITY_1_5,
                       judge=_LengthJudge())
    ev = await evaluate(study, progress=False)
    sheet = cafe.answer_sheet(ev)
    save_evaluation(ev, str(tmp_path / "ev.json"))
    back = load_evaluation(str(tmp_path / "ev.json"))

    assert {(type(o.config["top_k"]), o.config["top_k"]) for o in back.answers.observations} == {
        (int, 3), (int, 10)}
    assert [m["level"] for m in back.attribution.factor_marginals][:2] == ["3", "10"]
    assert back.attribution.factor_marginals == ev.attribution.factor_marginals
    assert back.records() == ev.records()                       # np.int64(3) == 3
    assert all(r["output"] is not None for r in back.records())  # each verdict joined
    scores = [{"answer_id": s["answer_id"], "rater": "h", "score": 3} for s in sheet]
    assert len(cafe.human_evaluation(back, scores).ratings.items) == len(sheet)


async def test_a_container_of_numpy_numbers_reads_the_same_after_a_reload(tmp_path):
    """A tuple level built from NumPy arrays holds NumPy numbers. Its label (JSON text) must
    show them as the numbers a saved evaluation brings back, "[256, 0]" rather than
    '["256", "0"]', so the reloaded evaluation reads as the original does."""
    np = pytest.importorskip("numpy")
    from cafe.evaluation import evaluate, load_evaluation, save_evaluation
    from cafe.execution.results import level_label

    chunking = list(zip(np.array([256, 512]), np.array([0, 64])))  # tuples of np.int64
    assert level_label(chunking[0]) == "[256, 0]"
    study = cafe.Study(name="np", system=lambda config, item: f"{config}:{item}",
                       factors=[cafe.Factor("chunking", chunking), cafe.Factor("m", ["a", "b"])],
                       dataset=[f"q{i}" for i in range(8)], rubric=cafe.ANSWER_QUALITY_1_5,
                       judge=_LengthJudge())
    ev = await evaluate(study, progress=False)
    save_evaluation(ev, str(tmp_path / "ev.json"))
    back = load_evaluation(str(tmp_path / "ev.json"))
    assert {m["level"] for m in back.attribution.factor_marginals} == {
        "[256, 0]", "[512, 64]", "a", "b"}
    assert back.attribution.factor_marginals == ev.attribution.factor_marginals
    assert back.show() == ev.show()


async def test_checkpoint_resume_finds_the_cells_of_numpy_levels(tmp_path):
    """A checkpoint writes levels as config_id encodes them, so a resumed run recognises every
    finished cell of a NumPy level and runs none of them again."""
    np = pytest.importorskip("numpy")
    from cafe.execution import run_study

    calls = []
    study = cafe.Study(name="np", system=lambda config, item: calls.append(item) or "x",
                       factors=[cafe.Factor("top_k", [np.int64(3), np.int64(10)])],
                       dataset=["q1", "q2"])
    ckpt = str(tmp_path / "run.jsonl")
    await run_study(study, checkpoint_path=ckpt)
    res = await run_study(study, checkpoint_path=ckpt)
    assert len(calls) == 4 and len(res) == 4


def test_a_bundle_saved_with_numpy_levels_as_text_still_loads():
    """Bundles saved before NumPy numbers were written as numbers hold such a level as text
    ("3"), with answer keys hashed from that text. They still load, every verdict joined to
    its answer (the level stays "3": its type was lost when it was saved)."""
    import hashlib

    from cafe.evaluation import load_evaluation

    old_key = hashlib.sha1(b'{"top_k": "3"}').hexdigest()[:12] + "::q0::0"
    rubric = cafe.ANSWER_QUALITY_1_5
    bundle = {
        "study_name": "old",
        "answers": {"factors": ["top_k"], "observations": [
            {"config": {"top_k": "3"}, "input_id": "q0", "rep": 0, "output": "x"}]},
        "ratings": {"factors": ["top_k"], "rubric": {
            "name": rubric.name, "scale_type": rubric.scale_type.value,
            "levels": [{"value": lv.value, "label": lv.label, "description": lv.description}
                       for lv in rubric.levels]},
            "items": [{"obs_key": old_key, "config": {"top_k": "3"}, "input_id": "q0", "rep": 0,
                       "judge_rep": 0, "value": 4, "value_numeric": 4}]},
    }
    (row,) = load_evaluation(bundle).records()
    assert (row["top_k"], row["output"], row["verdict"]) == ("3", "x", 4)
