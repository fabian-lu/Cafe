"""Judge prompt rendering — the preview must match exactly what is sent."""

import asyncio

import pytest

import cafe
from cafe.judging import structured as st


def test_render_messages_has_system_and_user():
    judge = cafe.LLMJudge(model="x", system_prompt="SYS-FRAMING")
    msgs = judge.render_messages(cafe.ANSWER_QUALITY_1_5, "Q?", "A.", reference="R")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == "SYS-FRAMING"
    # the user message is the rubric-derived prompt
    assert "Q?" in msgs[1]["content"] and "A." in msgs[1]["content"]


def test_judge_prompt_template_wins_over_rubric_and_preset():
    rubric = cafe.ANSWER_QUALITY_1_5
    judge = cafe.LLMJudge(model="x", preset="single_answer",
                          prompt_template="JUDGE-TPL {question} :: {answer} :: {scale} :: GRADE: <{grade}>")
    user = judge.render_messages(rubric, "Q?", "A.")[1]["content"]
    assert user.startswith("JUDGE-TPL Q? :: A. ::")
    assert "GRADE: <exactly one of: 1, 2, 3, 4, 5>" in user


@pytest.mark.parametrize("template", [
    'Q: {question} A: {answer} S: {scale} Reply as JSON like {"grade": {grade}}',
    "{questions} {answer} {scale} GRADE: <{grade}>",
    "{question} {answer} {scale} GRADE: <{grade}> }",
    "{question} {answer} {scale} {} GRADE: <{grade}>",
], ids=["json-example", "typo-placeholder", "stray-brace", "positional"])
def test_unrenderable_template_is_rejected_when_set(template):
    # used to pass construction and raise a bare KeyError/ValueError at the first judge call
    with pytest.raises(ValueError, match="judge prompt_template cannot be rendered.*doubled"):
        cafe.LLMJudge(model="m", prompt_template=template)
    with pytest.raises(ValueError, match="rubric 'r' prompt_template cannot be rendered"):
        cafe.Rubric(name="r", levels=cafe.ANSWER_QUALITY_1_5.levels, prompt_template=template)


def test_escaped_braces_render_literally_and_late_templates_fail_clearly():
    tpl = 'Q: {question} A: {answer} S: {scale} Reply as JSON like {{"grade": <{grade}>}}'
    user = cafe.LLMJudge(model="m", prompt_template=tpl).render_messages(
        cafe.ANSWER_QUALITY_1_5, "Q?", "A.")[1]["content"]
    assert user.endswith('Reply as JSON like {"grade": <exactly one of: 1, 2, 3, 4, 5>}')
    # a template assigned after construction skips the check; rendering still explains it
    judge = cafe.LLMJudge(model="m")
    judge.prompt_template = tpl.replace("{{", "{").replace("}}", "}")
    with pytest.raises(ValueError, match="judge prompt template cannot be rendered"):
        judge.render_messages(cafe.ANSWER_QUALITY_1_5, "Q?", "A.")


def test_builtin_rubrics_are_valid():
    assert set(cafe.rubrics.ALL) == {
        "answer_quality_1_5", "faithfulness_1_5", "relevance_1_5",
        "helpfulness_0_10", "correctness_0_3", "correct_pass_fail",
    }
    for r in cafe.rubrics.ALL.values():
        assert len(r.levels) >= 2
        assert r.numeric(r.max_value) == r.max_value          # round-trips on-scale
        assert r.numeric(r.max_value + 99) is None            # rejects off-scale
    assert cafe.rubrics.CORRECT_PASS_FAIL.scale_type == cafe.ScaleType.binary


@pytest.mark.parametrize(
    "levels",
    [
        [],
        [cafe.Level(1, "only", "the single level")],
        [cafe.Level(1, "ok", "x"), cafe.Level(1, "fine", "y")],  # two labels, one scale point
    ],
)
def test_rubric_needs_two_distinct_levels(levels):
    with pytest.raises(ValueError, match="at least two levels"):
        cafe.Rubric(name="degenerate", levels=levels)


def test_grade_hint_is_scale_aware():
    # numeric → a range (any integer valid); ordinal/binary → exact allowed values
    assert cafe.rubrics.HELPFULNESS_0_10.grade_hint() == "an integer from 0 to 10"
    assert cafe.rubrics.CORRECT_PASS_FAIL.grade_hint() == "exactly one of: 0, 1"
    assert cafe.ANSWER_QUALITY_1_5.grade_hint() == "exactly one of: 1, 2, 3, 4, 5"
    # a NON-contiguous ordinal scale lists exactly its levels, not a 1–5 range
    sparse = cafe.Rubric(name="s", scale_type=cafe.ScaleType.ordinal,
                         levels=[cafe.Level(1, "a", "x"), cafe.Level(3, "b", "y"), cafe.Level(5, "c", "z")])
    assert sparse.grade_hint() == "exactly one of: 1, 3, 5"


def test_preset_grade_line_matches_scale():
    j = cafe.LLMJudge(model="m")
    assert "GRADE: <an integer from 0 to 10>" in j.preview(cafe.rubrics.HELPFULNESS_0_10, "Q?", "A.")
    assert "GRADE: <exactly one of: 0, 1>" in j.preview(cafe.rubrics.CORRECT_PASS_FAIL, "Q?", "A.", reference="r")


def test_numeric_scale_accepts_any_in_range_between_anchors():
    r = cafe.rubrics.HELPFULNESS_0_10
    assert r.scale_type == cafe.ScaleType.numeric
    assert r.numeric(7) == 7        # 7 is in [0, 10] but not a defined anchor level
    assert r.numeric(0) == 0 and r.numeric(10) == 10
    assert r.numeric(11) is None and r.numeric(-1) is None
    # ordinal scales still require an exact level match
    assert cafe.rubrics.ANSWER_QUALITY_1_5.numeric(3) == 3


def test_parse_verdict_accepts_the_displayed_grade_delimiters():
    # every preset ends with `GRADE: <{grade}>`, so a judge copying it writes `GRADE: <4>`
    parse, r = cafe.judging.parse_verdict, cafe.ANSWER_QUALITY_1_5
    for raw in ("Fine.\nGRADE: <4>", "Fine.\nGRADE: [4]", "Fine.\nGRADE: 4",
                "Fine.\nGRADE: 4.", "Fine. [[4]]"):
        assert parse(raw, r) == (4, 4, "Fine."), raw
    assert parse("GRADE: <2> at first, but\nGRADE: <5>", r)[:2] == (5, 5)  # last marker wins
    likert = cafe.Rubric(name="likert", levels=[cafe.Level(v, str(v), "") for v in range(-2, 3)])
    assert parse("Meh.\nGRADE: <-1>", likert)[:2] == (-1, -1)


def test_fractional_grades_are_rejected_not_truncated():
    parse, r = cafe.judging.parse_verdict, cafe.ANSWER_QUALITY_1_5
    for raw in ("Mostly right.\nGRADE: 4.5", "Mostly right.\nGRADE: <4.5>", "Mostly. [[4.5]]"):
        value, numeric, why = parse(raw, r)
        assert (value, numeric) == (4.5, None), raw
        assert "not on scale" in why
    assert parse("ok\nGRADE: 7.8", cafe.rubrics.HELPFULNESS_0_10)[:2] == (7.8, None)
    assert parse("ok\nGRADE: 4.0", r)[:2] == (4, 4)  # an integral decimal is that integer
    # Rubric.numeric is the shared mapping (the JSON verdict path goes through it too)
    assert r.numeric(4.9) is None and r.numeric(4.0) == 4
    assert r.numeric(float("inf")) is None and r.numeric(float("nan")) is None
    assert cafe.rubrics.HELPFULNESS_0_10.numeric(7.8) is None
    v, n, _ = st.parse_json_verdict('{"reasoning": "x", "grade": 4.5}', r)
    assert (v, n) == (4.5, None)


def test_preview_includes_system_and_user():
    j = cafe.LLMJudge(model="m", system_prompt="You are terse.")
    out = j.preview(cafe.ANSWER_QUALITY_1_5, "Q?", "A.", reference="ref")
    assert out.startswith("[SYSTEM]\nYou are terse.")
    assert "[USER]" in out
    # the user block equals the render_messages user message (system is the only addition)
    assert j.render_messages(cafe.ANSWER_QUALITY_1_5, "Q?", "A.", reference="ref")[1]["content"] in out


def test_numeric_rubric_prompt_shows_anchors_and_full_range():
    out = cafe.LLMJudge(model="m").preview(cafe.rubrics.HELPFULNESS_0_10, "Q?", "A.")
    assert "0 = useless" in out and "10 = ideal" in out   # anchor levels
    assert "an integer from 0 to 10" in out               # judge told the full range


def test_warns_when_reference_ignored_by_template():
    j = cafe.LLMJudge(model="m", preset="single_answer")  # reference-free: no {reference}
    with pytest.warns(UserWarning, match="reference is being IGNORED"):
        j.render_messages(cafe.ANSWER_QUALITY_1_5, "Q?", "A.", reference="gold")


def test_no_warning_when_reference_used_or_absent():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        # reference-guided template uses {reference} → reference is honoured, no warning
        cafe.LLMJudge(model="m").render_messages(cafe.ANSWER_QUALITY_1_5, "Q?", "A.", reference="gold")
        # no reference passed → nothing to ignore, no warning
        cafe.LLMJudge(model="m", preset="single_answer").render_messages(
            cafe.ANSWER_QUALITY_1_5, "Q?", "A.")


def test_parse_json_verdict_ok_and_fences():
    r = cafe.ANSWER_QUALITY_1_5
    v, n, why = st.parse_json_verdict('{"reasoning": "good", "grade": 4}', r)
    assert (v, n, why) == (4, 4, "good")
    # tolerates a code fence / surrounding prose
    v, n, _ = st.parse_json_verdict('```json\n{"grade": 5}\n```', r)
    assert n == 5


def test_parse_json_verdict_ignores_braces_in_surrounding_prose():
    r = cafe.ANSWER_QUALITY_1_5
    for raw in ('{"reasoning": "ok", "grade": 4} Note: the {grade} key is set.',
                '{"reasoning": "ok", "grade": 4}\n\n(Scale used: {1..5})',
                'Using the {reasoning, grade} format: {"reasoning": "ok", "grade": 4}',
                '{"reasoning": {"steps": ["a"]}, "grade": 4} {done}'):
        v, n, _ = st.parse_json_verdict(raw, r)
        assert (v, n) == (4, 4), raw
    # two complete objects are ambiguous: not guessed, so the caller's fallback decides
    two = '{"reasoning": "draft", "grade": 2}\n{"reasoning": "final", "grade": 4}'
    assert st.parse_json_verdict(two, r) == (None, None, None)


def test_structured_score_keeps_json_verdict_followed_by_prose(monkeypatch):
    async def fake_complete(model, messages, **kwargs):
        return '{"reasoning": "Correct and complete.", "grade": 4} (as {"grade": n})'

    monkeypatch.setattr("cafe.judging.judge.complete", fake_complete)
    judge = cafe.LLMJudge("m", structured=True)
    out = asyncio.run(judge.score(cafe.ANSWER_QUALITY_1_5, "Q?", "A."))
    assert (out.value, out.value_numeric, out.reasoning) == (4, 4, "Correct and complete.")


def test_parse_json_verdict_failures_signal_fallback():
    r = cafe.ANSWER_QUALITY_1_5
    assert st.parse_json_verdict("not json at all", r) == (None, None, None)   # → regex fallback
    assert st.parse_json_verdict('{"reasoning": "x"}', r) == (None, None, None)  # no grade key
    # off-scale → an error verdict that keeps the grade (not the fallback sentinel)
    assert st.parse_json_verdict('{"grade": 99}', r) == (99, None, "verdict 99 not on scale 1-5")


def test_structured_off_scale_grade_is_not_reparsed_from_the_reasoning(monkeypatch):
    replies = {
        # the judge's grade is 7 (off a 1-5 scale); "grade: 3" only appears in its reasoning
        "fabricated": '{"reasoning": "strictly I would give grade: 3, but overall", "grade": 7}',
        "plain": '{"reasoning": "far better than the scale allows", "grade": 7}',
        "not_a_number": '{"reasoning": "between good and excellent", "grade": "4-5"}',
        "not_json": "The answer is fine.\nGRADE: 4",  # genuinely non-JSON → regex fallback
    }

    def score(kind):
        async def fake_complete(model, messages, **kwargs):
            return replies[kind]

        monkeypatch.setattr("cafe.judging.judge.complete", fake_complete)
        judge = cafe.LLMJudge("m", structured=True)
        out = asyncio.run(judge.score(cafe.ANSWER_QUALITY_1_5, "Q", "A"))
        return out.value, out.value_numeric, out.reasoning

    assert score("fabricated") == (7, None, "verdict 7 not on scale 1-5")
    assert score("plain") == (7, None, "verdict 7 not on scale 1-5")
    assert score("not_a_number") == ("4-5", None, "verdict '4-5' not on scale 1-5")
    assert score("not_json") == (4, 4, "The answer is fine.")


def test_structured_forced_decisions_need_no_llm():
    assert asyncio.run(cafe.LLMJudge("m", structured=False).prepare()) is False
    assert asyncio.run(cafe.LLMJudge("m", structured=True).prepare()) is True


def test_structured_auto_uses_cached_capability(monkeypatch):
    monkeypatch.setitem(st._support_cache, "fake-model", True)   # pretend probe already ran
    assert asyncio.run(cafe.LLMJudge("fake-model", structured="auto").prepare()) is True


def test_structured_validates():
    with pytest.raises(ValueError, match="structured"):
        cafe.LLMJudge("m", structured="sometimes")


def test_preview_shows_both_messages():
    study = cafe.Study(
        name="t",
        system=lambda c, i: "x",
        dataset=[{"text": "Can water turn into wine?", "reference": "No."}],
        rubric=cafe.ANSWER_QUALITY_1_5,
        judge=cafe.LLMJudge(model="x", system_prompt="You are a strict evaluator."),
    )
    preview = study.preview_judge_prompt(answer="Yes, trivially.")
    assert "[SYSTEM]" in preview and "[USER]" in preview
    assert "You are a strict evaluator." in preview
    assert "Can water turn into wine?" in preview
