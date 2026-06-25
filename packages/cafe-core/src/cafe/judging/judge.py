"""The judge: a :class:`Judge` protocol and the batteries-included :class:`LLMJudge`.

Implement :class:`Judge` for a custom or non-LLM grader; :class:`LLMJudge` scores
with any LiteLLM model, using a research-grounded prompt preset (see
:mod:`cafe.judging.prompts`).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cafe.judging.prompts import JUDGE_PRESETS, build_judge_prompt, parse_verdict
from cafe.judging.ratings import JudgeOutput
from cafe.judging.rubric import Rubric
from cafe.llm import LLMError, complete


@runtime_checkable
class Judge(Protocol):
    """Anything that can score an answer. Implement this for a custom/non-LLM judge."""

    model: str

    async def score(
        self, rubric: Rubric, question: str, answer: str, reference: str | None = ...
    ) -> JudgeOutput: ...


class LLMJudge:
    """Scores answers with an LLM (any provider, via a LiteLLM model string)."""

    #: System message framing the judge; override per instance if you like.
    default_system = "You are a strict, fair, impartial evaluator."

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        preset: str = "reference_guided",
        system_prompt: str | None = None,
    ) -> None:
        if preset not in JUDGE_PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(JUDGE_PRESETS)}")
        self.model = model
        self.temperature = temperature
        self.preset = preset
        self.system_prompt = system_prompt or self.default_system

    def render_prompt(
        self, rubric: Rubric, question: str, answer: str, reference: str | None = None
    ) -> str:
        """The exact prompt this judge would send — without calling the LLM."""
        return build_judge_prompt(rubric, question, answer, reference, preset=self.preset)

    async def score(
        self, rubric: Rubric, question: str, answer: str, reference: str | None = None
    ) -> JudgeOutput:
        prompt = self.render_prompt(rubric, question, answer, reference)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = await complete(self.model, messages, temperature=self.temperature)
        except LLMError as exc:
            return JudgeOutput(None, None, f"judge call failed: {exc}", prompt, None)
        value, numeric, reasoning = parse_verdict(raw, rubric)
        return JudgeOutput(value, numeric, reasoning, prompt, raw)
