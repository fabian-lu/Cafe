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


def test_parse_json_verdict_failures_signal_fallback():
    r = cafe.ANSWER_QUALITY_1_5
    assert st.parse_json_verdict("not json at all", r) == (None, None, None)   # → regex fallback
    assert st.parse_json_verdict('{"reasoning": "x"}', r) == (None, None, None)  # no grade key
    v, n, _ = st.parse_json_verdict('{"grade": 99}', r)
    assert (v, n) == (99, None)  # off-scale → numeric None (caller still falls back)


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


def test_parse_verdict_accepts_the_delimiters_the_prompt_displays():
    from cafe.judging.prompts import parse_verdict

    r = cafe.ANSWER_QUALITY_1_5
    # the templates literally show `GRADE: <...>` — a judge mimicking them must parse
    assert parse_verdict("Correct answer.\nGRADE: <4>", r)[:2] == (4, 4)
    assert parse_verdict("ok\nGRADE: [4]", r)[:2] == (4, 4)      # bracket form still works
    assert parse_verdict("ok\nGRADE: 4", r)[:2] == (4, 4)        # bare form still works


def test_parse_verdict_rejects_fractional_grades():
    from cafe.judging.prompts import parse_verdict

    v, n, why = parse_verdict("hmm\nGRADE: 4.5", cafe.ANSWER_QUALITY_1_5)
    assert v == 4.5 and n is None                # off-scale error, not a silent floor to 4
    assert "not on scale" in why


def test_rubric_numeric_rejects_fractional_values():
    assert cafe.rubrics.HELPFULNESS_0_10.numeric(4.9) is None    # never truncated to 4
    assert cafe.rubrics.HELPFULNESS_0_10.numeric(4.0) == 4       # integer-valued float ok
    assert cafe.ANSWER_QUALITY_1_5.numeric(4.5) is None


def test_parse_json_verdict_survives_trailing_prose_with_braces():
    r = cafe.ANSWER_QUALITY_1_5
    raw = '{"reasoning": "ok", "grade": 4} Note: the {grade} key is set.'
    assert st.parse_json_verdict(raw, r) == (4, 4, "ok")
