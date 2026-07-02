"""A small library of ready-made rubrics — grab one, or copy it as a template.

Each is a plain :class:`cafe.Rubric`; nothing here is special. Use them directly::

    import cafe
    study = cafe.Study(..., rubric=cafe.rubrics.FAITHFULNESS_1_5, judge=...)

or copy one and edit the levels/instruction for your own criterion. Pick the
``scale_type`` deliberately — it decides which statistical model runs (ordinal →
CLMM, numeric → linear, binary → logistic).
"""

from __future__ import annotations

from cafe.judging.rubric import ANSWER_QUALITY_1_5, Level, Rubric, ScaleType

# Reference-guided correctness/helpfulness on a 1–5 ordinal scale (the default).
# Re-exported so everything lives under one namespace.
ANSWER_QUALITY_1_5 = ANSWER_QUALITY_1_5

# How well the answer is supported by the reference/source (groundedness).
FAITHFULNESS_1_5 = Rubric(
    name="faithfulness_1_5",
    scale_type=ScaleType.ordinal,
    levels=[
        Level(1, "contradicted", "Contradicts the reference / source."),
        Level(2, "unsupported", "Largely unsupported or invents claims."),
        Level(3, "mixed", "A mix of supported and unsupported content."),
        Level(4, "mostly_grounded", "Supported by the reference, minor slips."),
        Level(5, "fully_grounded", "Every claim is supported by the reference."),
    ],
    instruction="Judge how faithfully the ANSWER is supported by the REFERENCE — penalise unsupported claims.",
)

# Reference-free helpfulness/relevance (pair with the mtbench_single preset).
RELEVANCE_1_5 = Rubric(
    name="relevance_1_5",
    scale_type=ScaleType.ordinal,
    levels=[
        Level(1, "off_topic", "Does not address the question."),
        Level(2, "weak", "Barely addresses it; mostly unhelpful."),
        Level(3, "partial", "Addresses part of it, with gaps."),
        Level(4, "good", "Addresses it helpfully, minor gaps."),
        Level(5, "excellent", "Directly, completely, and helpfully addresses it."),
    ],
    instruction="Judge how relevant and helpful the ANSWER is to the QUESTION (ignore correctness of unverifiable facts).",
)

# A 0–10 numeric (interval) helpfulness score — the levels are anchors, and the judge
# may return any integer in range. `scale_type=numeric` runs the linear model.
HELPFULNESS_0_10 = Rubric(
    name="helpfulness_0_10",
    scale_type=ScaleType.numeric,
    levels=[
        Level(0, "useless", "No help at all — wrong, empty, or off-topic."),
        Level(5, "partial", "Somewhat helpful, with notable gaps or errors."),
        Level(10, "ideal", "Complete, correct, and maximally helpful."),
    ],
    instruction="Rate how helpful the ANSWER is to the QUESTION on a 0–10 scale (0=useless, 10=ideal).",
)

# A two-outcome pass/fail (logistic stats). 1 = pass, 0 = fail.
CORRECT_PASS_FAIL = Rubric(
    name="correct_pass_fail",
    scale_type=ScaleType.binary,
    levels=[
        Level(0, "fail", "Incorrect, misleading, or unsupported."),
        Level(1, "pass", "Correct and adequately supported."),
    ],
    instruction="Decide whether the ANSWER is correct and adequately supported.",
)

#: Everything in the library, by name — handy for listing / a UI dropdown.
ALL: dict[str, Rubric] = {
    r.name: r
    for r in (ANSWER_QUALITY_1_5, FAITHFULNESS_1_5, RELEVANCE_1_5,
              HELPFULNESS_0_10, CORRECT_PASS_FAIL)
}

__all__ = [
    "ANSWER_QUALITY_1_5",
    "FAITHFULNESS_1_5",
    "RELEVANCE_1_5",
    "HELPFULNESS_0_10",
    "CORRECT_PASS_FAIL",
    "ALL",
]
