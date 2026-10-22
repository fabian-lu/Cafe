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
# The presets display the marker as `GRADE: <...>`, so `<4>` (and `[4]`) are accepted. The
# whole number is captured — sign and fractional part included — so `GRADE: 4.5` is
# rejected as off-scale instead of being truncated to 4, and negative levels parse.
_GRADE_RE = re.compile(r"(?is)GRADE\s*:\s*[<\[]?\s*(-?[0-9]+(?:\.[0-9]+)?)")
_BRACKET_RE = re.compile(r"\[\[\s*(-?[0-9]+(?:\.[0-9]+)?)\s*\]\]")

#: Placeholders a custom judge ``prompt_template`` must contain to be a valid grading prompt.
REQUIRED_PLACEHOLDERS = ("{question}", "{answer}", "{scale}", "{grade}")

# Stand-ins for every placeholder build_judge_prompt fills (same types: text, int min/max),
# used to trial-render a custom template as soon as it is set.
_TRIAL_VALUES: dict[str, Any] = dict(
    instruction="x", question="x", answer="x", reference="x", scale="x", grade="x", min=0, max=1
)
_TEMPLATE_HELP = (
    "Placeholders are {instruction} {question} {answer} {reference} {scale} {grade} {min} "
    "{max}; any other brace must be doubled to appear literally, e.g. a JSON example is "
    'written {{"grade": 4}}.'
)


def _render(template: str, values: dict[str, Any], where: str) -> str:
    """``template.format(**values)``, with a format failure turned into an actionable error."""
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError, AttributeError, TypeError) as exc:
        raise ValueError(
            f"{where} cannot be rendered ({type(exc).__name__}: {exc}). {_TEMPLATE_HELP}"
        ) from exc


def check_template_placeholders(template: str, *, where: str = "prompt_template") -> None:
    """Validate a custom judge ``template``. Called once when the template is set (on the
    Rubric or the LLMJudge), so a broken template fails when the study is defined — not
    at the first judge call, after the answers were already generated and paid for.

    Raises
    ------
    ValueError
        If the template cannot be rendered: an unknown placeholder (e.g. ``{questions}``)
        or an unescaped literal brace — a JSON example must be written ``{{"grade": 4}}``.

    Warns
    -----
    UserWarning
        If a placeholder needed for a valid grading prompt is missing — e.g. a template
        with no ``{answer}`` would ask the judge to grade nothing.
    """
    _render(template, _TRIAL_VALUES, where)
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

    Raises
    ------
    ValueError
        If the chosen template cannot be rendered (an unknown placeholder or an unescaped
        literal brace). Templates set on a Rubric or LLMJudge are already checked when set.
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
    values = dict(
        instruction=rubric.instruction,
        question=question,
        answer=answer,
        reference=reference if reference else "(no reference provided)",
        scale=rubric.scale_text(),
        grade=rubric.grade_hint(),   # scale-aware: range for numeric, exact values for ordinal/binary
        min=rubric.min_value,
        max=rubric.max_value,
    )
    return _render(chosen, values, "judge prompt template")


def parse_verdict(raw: str, rubric: Rubric) -> tuple[Any, int | None, str | None]:
    """Extract (raw_value, numeric, reasoning) from a reason-then-GRADE response.

    The last ``GRADE: <n>`` marker wins (``GRADE: n``, ``GRADE: [n]`` and an MT-Bench
    style ``[[n]]`` are accepted too).

    Returns
    -------
    tuple
        ``(value, numeric, reasoning)``. On success ``numeric`` is the rubric scale point
        and ``reasoning`` the prose before the marker. Otherwise ``numeric`` is ``None``
        and the third element is the error message. Grades must be whole numbers on the
        rubric's scale: a fractional grade (``GRADE: 4.5``) is kept as ``value`` but
        rejected like any off-scale grade — never rounded or truncated. An integral
        decimal (``GRADE: 4.0``) counts as 4.
    """
    if not raw or not raw.strip():
        return None, None, "empty judge response"
    matches = list(_GRADE_RE.finditer(raw)) or list(_BRACKET_RE.finditer(raw))
    if not matches:
        return None, None, f"no GRADE marker in judge output: {raw[:120]!r}"
    last = matches[-1]
    number = float(last.group(1))
    value: Any = int(number) if number.is_integer() else number  # 4.5 stays 4.5 (off-scale)
    reasoning = raw[: last.start()].strip() or None  # the prose before the verdict
    numeric = rubric.numeric(value)
    if numeric is None:
        return value, None, f"verdict {value} not on scale {rubric.min_value}-{rubric.max_value}"
    return value, numeric, reasoning
