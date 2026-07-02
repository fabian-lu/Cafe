"""Judge prompt presets and verdict parsing.

All presets ask the judge to reason in prose and then emit a single ``GRADE: <int>``
line, so parsing is uniform (we capture the *last* marker, the MT-Bench / Inspect
convention). Wording is adapted from published judges and cited below; swap via
``LLMJudge(preset=...)`` or override the whole prompt with ``rubric.prompt_template``.
"""

from __future__ import annotations

import re
from typing import Any

from cafe.judging.rubric import Rubric

# Adapted from MT-Bench's reference-guided "single-math-v1" grader
# (Zheng et al., 2023, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena")
# and Inspect AI's model_graded_qa (reason-then-grade with a parseable marker).
_REFERENCE_GUIDED = """\
Please act as an impartial judge. {instruction}

[BEGIN DATA]
***
[Question]: {question}
***
[Answer]: {answer}
***
[Reference answer]: {reference}
***
[END DATA]

Begin your evaluation by reasoning step by step: compare the Answer against the
Reference answer, identify and correct any mistakes, and note unsupported claims.
Judge correctness and helpfulness — not style, length, or formatting.

Score on this scale:
{scale}

Then finish with exactly one final line, and nothing after it:
GRADE: <{grade}>"""

# Adapted from MT-Bench "single-v1" (reference-free single-answer grading).
_MTBENCH_SINGLE = """\
[Instruction] Please act as an impartial judge and evaluate the quality of the
ANSWER provided to the QUESTION below. {instruction} Consider helpfulness,
relevance, accuracy, and depth. Be as objective as possible.

[Question]
{question}

[Answer]
{answer}

Begin with a short explanation, then score on this scale:
{scale}

Finish with exactly one final line: GRADE: <{grade}>"""

# Adapted from Inspect AI's model_graded_qa criterion grader.
_CRITERION_GRADED = """\
You are assessing a submitted answer against a criterion.

[BEGIN DATA]
***
[Task]: {question}
***
[Submission]: {answer}
***
[Reference]: {reference}
***
[Criterion]: {instruction}
***
[END DATA]

Write out, step by step, your reasoning about how well the submission meets the
criterion (do not just state the answer). Then score on this scale:
{scale}

End with exactly one final line: GRADE: <{grade}>"""

JUDGE_PRESETS: dict[str, str] = {
    "reference_guided": _REFERENCE_GUIDED,
    "mtbench_single": _MTBENCH_SINGLE,
    "criterion_graded": _CRITERION_GRADED,
}

# Greedy: capture the LAST marker, so grades mentioned mid-reasoning don't fool us.
_GRADE_RE = re.compile(r"(?is)GRADE\s*:\s*\[?\[?\s*([0-9]+)")
_BRACKET_RE = re.compile(r"\[\[\s*([0-9]+)\s*\]\]")


def build_judge_prompt(
    rubric: Rubric,
    question: str,
    answer: str,
    reference: str | None = None,
    *,
    preset: str = "reference_guided",
    template: str | None = None,
) -> str:
    """Render the exact prompt sent to the judge.

    Precedence for the template: an explicit ``template`` (the judge's own) →
    ``rubric.prompt_template`` → the named ``preset``.
    """
    chosen = template or rubric.prompt_template or JUDGE_PRESETS.get(preset, _REFERENCE_GUIDED)
    if reference and "{reference}" not in chosen:
        import warnings

        # Silently dropping a gold answer would look like reference-guided judging while
        # actually grading reference-free — a validity trap. Warn once (Python dedupes).
        warnings.warn(
            "a reference was provided but the judge prompt has no {reference} placeholder "
            "(e.g. the reference-free 'mtbench_single' preset, or a custom template missing "
            "it) — the reference is being IGNORED. Use a reference-guided preset "
            "(reference_guided / criterion_graded) or add {reference} to your template.",
            stacklevel=2,
        )
    return chosen.format(
        instruction=rubric.instruction,
        question=question,
        answer=answer,
        reference=reference if reference else "(no reference provided)",
        scale=rubric.scale_text(),
        grade=rubric.grade_hint(),   # scale-aware: range for numeric, exact values for ordinal/binary
        min=rubric.min_value,
        max=rubric.max_value,
    )


def parse_verdict(raw: str, rubric: Rubric) -> tuple[Any, int | None, str | None]:
    """Extract (raw_value, numeric, reasoning) from a reason-then-GRADE response."""
    if not raw or not raw.strip():
        return None, None, "empty judge response"
    matches = list(_GRADE_RE.finditer(raw)) or list(_BRACKET_RE.finditer(raw))
    if not matches:
        return None, None, f"no GRADE marker in judge output: {raw[:120]!r}"
    last = matches[-1]
    value = int(last.group(1))
    reasoning = raw[: last.start()].strip() or None  # the prose before the verdict
    numeric = rubric.numeric(value)
    if numeric is None:
        return value, None, f"verdict {value} not on scale {rubric.min_value}-{rubric.max_value}"
    return value, numeric, reasoning
