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
# NB: what to reward/ignore (style, length, tone) belongs to the *rubric's* instruction,
# not this shared template — a rubric may deliberately evaluate tone or length.
_REFERENCE_QA = """\
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

Score on this scale:
{scale}

Then finish with exactly one final line, and nothing after it:
GRADE: <{grade}>"""

# Adapted from MT-Bench "single-v1" (reference-free single-answer grading).
_SINGLE_ANSWER = """\
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
_CRITERION = """\
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
    "reference_qa": _REFERENCE_QA,          # reference-guided QA (MT-Bench single-math / Inspect)
    "single_answer": _SINGLE_ANSWER,        # reference-free single-answer (MT-Bench single-v1)
    "criterion": _CRITERION,                # assess against a stated criterion (Inspect model_graded_qa)
}

# Greedy: capture the LAST marker, so grades mentioned mid-reasoning don't fool us.
_GRADE_RE = re.compile(r"(?is)GRADE\s*:\s*\[?\s*([0-9]+)")
_BRACKET_RE = re.compile(r"\[\[\s*([0-9]+)\s*\]\]")

#: Placeholders a custom judge ``prompt_template`` must contain to be a valid grading prompt.
REQUIRED_PLACEHOLDERS = ("{question}", "{answer}", "{scale}", "{grade}")


def check_template_placeholders(template: str, *, where: str = "prompt_template") -> None:
    """Warn if a custom judge ``template`` is missing a placeholder needed for a valid
    grading prompt — e.g. a template with no ``{answer}`` would ask the judge to grade
    nothing. Called once when the template is set (on the Rubric or the LLMJudge)."""
    missing = [p for p in REQUIRED_PLACEHOLDERS if p not in template]
    if missing:
        import warnings

        warnings.warn(
            f"{where} is missing placeholder(s) {missing} — the judge won't see "
            f"{'/'.join(m.strip('{}') for m in missing)}. Include them, or the grade may be "
            "meaningless. (Optional: {reference} for reference-guided grading.)",
            stacklevel=3,
        )


def build_judge_prompt(
    rubric: Rubric,
    question: str,
    answer: str,
    reference: str | None = None,
    *,
    preset: str = "reference_qa",
    template: str | None = None,
) -> str:
    """Render the exact prompt sent to the judge.

    Precedence for the template: an explicit ``template`` (the judge's own) →
    ``rubric.prompt_template`` → the named ``preset``.
    """
    chosen = template or rubric.prompt_template or JUDGE_PRESETS.get(preset, _REFERENCE_QA)
    if reference and "{reference}" not in chosen:
        import warnings

        # Silently dropping a gold answer would look like reference-guided judging while
        # actually grading reference-free — a validity trap. Warn once (Python dedupes).
        warnings.warn(
            "a reference was provided but the judge prompt has no {reference} placeholder "
            "(e.g. the reference-free 'single_answer' preset, or a custom template missing "
            "it) — the reference is being IGNORED. Use a reference-guided preset "
            "(reference_qa / criterion) or add {reference} to your template.",
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
