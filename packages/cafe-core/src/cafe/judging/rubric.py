"""Rubrics: the scale a judge (LLM or human) scores answers on, and how the judge
is prompted.

A rubric is an ordered list of levels plus a ``scale_type`` that tells the
statistics layer how to treat the numbers (ordinal → cumulative-link model;
numeric → linear model; binary → logistic). The order of levels defines the
numeric mapping. Keep scales short and well-described — the judge reads them.

The judge *prompt* is assembled from the rubric by :mod:`cafe.judging`, using a
research-grounded default (MT-Bench / Inspect style) unless ``prompt_template`` is
set. Either way the full prompt is printable and editable — never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScaleType(str, Enum):
    """How the rubric's numbers should be interpreted (drives the statistics)."""

    ordinal = "ordinal"    # ordered categories, unequal spacing (e.g. 1..5 quality)
    numeric = "numeric"    # interval/ratio score (e.g. 0..10 treated as continuous)
    binary = "binary"      # two outcomes (pass/fail, yes/no)


@dataclass(frozen=True)
class Level:
    """One point on a rubric's ordered scale."""

    value: int
    label: str
    description: str


@dataclass
class Rubric:
    """An ordered quality scale + how the judge is prompted.

    Set ``prompt_template`` to take full control of the judge prompt (placeholders:
    ``{instruction}``, ``{question}``, ``{answer}``, ``{reference}``, ``{scale}``,
    ``{min}``, ``{max}``, ``{grade}`` — the scale-aware "allowed final grade" hint).
    Leave it ``None`` to use the cited default in :mod:`cafe.judging`.
    """

    name: str
    levels: list[Level]
    scale_type: ScaleType = ScaleType.ordinal
    instruction: str = "Judge how well the ANSWER responds to the QUESTION."
    prompt_template: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.scale_type, str):
            self.scale_type = ScaleType(self.scale_type)
        if not self.levels:
            raise ValueError("a rubric needs at least two levels")
        if self.prompt_template is not None:
            from cafe.judging.prompts import check_template_placeholders

            check_template_placeholders(self.prompt_template, where=f"rubric {self.name!r} prompt_template")

    def numeric(self, value: object) -> int | None:
        """Map a raw verdict value onto its integer scale point (or None).

        For ``numeric`` scales any integer within ``[min_value, max_value]`` is valid
        (the levels are just anchors); for ordinal/binary the value must match a defined
        level exactly."""
        try:
            iv = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if self.scale_type == ScaleType.numeric:
            return iv if self.min_value <= iv <= self.max_value else None
        return iv if any(lvl.value == iv for lvl in self.levels) else None

    def scale_text(self) -> str:
        """A compact, judge-readable description of the scale."""
        return "\n".join(f"  {lvl.value} = {lvl.label}: {lvl.description}" for lvl in self.levels)

    def grade_hint(self) -> str:
        """How to constrain the judge's final grade, matched to the scale type: a range
        for ``numeric`` (any integer in it is valid — the levels are anchors), or the
        **exact allowed values** for ``ordinal``/``binary`` (only the defined levels are
        valid, and they may be non-contiguous, e.g. 1, 3, 5)."""
        if self.scale_type == ScaleType.numeric:
            return f"an integer from {self.min_value} to {self.max_value}"
        return "exactly one of: " + ", ".join(str(lvl.value) for lvl in self.levels)

    @property
    def min_value(self) -> int:
        return min(lvl.value for lvl in self.levels)

    @property
    def max_value(self) -> int:
        return max(lvl.value for lvl in self.levels)


# A reasonable default: 1..5 ordinal answer-quality scale.
ANSWER_QUALITY_1_5 = Rubric(
    name="answer_quality_1_5",
    scale_type=ScaleType.ordinal,
    levels=[
        Level(1, "wrong", "Incorrect, irrelevant, or misleading."),
        Level(2, "weak", "Mostly unhelpful or substantially inaccurate."),
        Level(3, "ok", "Partially correct and somewhat helpful, with gaps."),
        Level(4, "good", "Correct and helpful, minor issues at most."),
        Level(5, "excellent", "Correct, complete, and clearly helpful."),
    ],
    instruction=("Judge the correctness and helpfulness of the ANSWER to the QUESTION. "
                 "Reward substance, not style — do not prefer an answer for being longer, "
                 "more formatted, or matching the reference's wording."),
)
