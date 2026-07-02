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
        prompt_template: str | None = None,
        structured: str | bool = "auto",
    ) -> None:
        if preset not in JUDGE_PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(JUDGE_PRESETS)}")
        if structured not in (True, False, "auto"):
            raise ValueError(f"structured must be True, False, or 'auto'; got {structured!r}")
        self.model = model
        self.temperature = temperature
        self.preset = preset
        self.system_prompt = system_prompt or self.default_system
        #: Full override of the user prompt (placeholders: {instruction} {question}
        #: {answer} {reference} {scale} {grade} {min} {max}). Wins over the rubric's
        #: template and the preset. Must still elicit a final ``GRADE: <int>`` line.
        self.prompt_template = prompt_template
        #: Ask for a JSON verdict instead of a GRADE: line — ``True``/``False`` to force,
        #: ``"auto"`` to use it where the model supports it (static map, else one probe).
        self.structured = structured
        self._use_json: bool | None = None  # resolved once by prepare()/score()

    def render_messages(
        self, rubric: Rubric, question: str, answer: str, reference: str | None = None
    ) -> list[dict[str, str]]:
        """The **exact** message list sent to the judge — the system framing plus the
        rubric-derived user prompt. This is the single source of truth shared with
        :meth:`score`, so a preview can never drift from what's actually sent. For a
        human-readable view, use :meth:`preview`."""
        user = build_judge_prompt(
            rubric, question, answer, reference,
            preset=self.preset, template=self.prompt_template,
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

    def preview(
        self, rubric: Rubric, question: str, answer: str, reference: str | None = None
    ) -> str:
        """The **full** judge input exactly as sent — the system message *and* the user
        prompt, labelled ``[SYSTEM]`` / ``[USER]`` (no LLM call). This is the way to see
        everything the judge receives."""
        return "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}"
            for m in self.render_messages(rubric, question, answer, reference)
        )

    async def prepare(self) -> bool:
        """Resolve one-time setup before scoring — here, whether to use structured (JSON)
        output for this model. Idempotent and cached; the judging runner calls it **once**
        before the loop so the capability probe never runs per-call."""
        if self._use_json is None:
            if self.structured in (True, False):
                self._use_json = bool(self.structured)
            else:  # "auto"
                from cafe.judging.structured import supports_structured

                self._use_json = await supports_structured(self.model)
        return self._use_json

    async def score(
        self, rubric: Rubric, question: str, answer: str, reference: str | None = None
    ) -> JudgeOutput:
        messages = self.render_messages(rubric, question, answer, reference)
        prompt = messages[-1]["content"]

        if await self.prepare():
            out = await self._score_structured(rubric, messages, prompt)
            if out is not None:  # None ⇒ provider rejected JSON; fall back to a plain call
                return out

        try:
            raw = await complete(self.model, messages, temperature=self.temperature)
        except LLMError as exc:
            return JudgeOutput(None, None, f"judge call failed: {exc}", prompt, None)
        value, numeric, reasoning = parse_verdict(raw, rubric)
        return JudgeOutput(value, numeric, reasoning, prompt, raw)

    async def _score_structured(
        self, rubric: Rubric, messages: list[dict[str, str]], prompt: str
    ) -> JudgeOutput | None:
        """Score by asking for a JSON verdict. Returns ``None`` only if the provider
        rejects ``response_format`` (so the caller falls back to a plain call); a parsed
        result — even an unparseable one — is returned with its raw response for audit."""
        from cafe.judging.structured import JSON_INSTRUCTION, parse_json_verdict

        json_prompt = prompt + JSON_INSTRUCTION.format(grade=rubric.grade_hint())
        json_messages = [messages[0], {"role": "user", "content": json_prompt}]
        try:
            raw = await complete(
                self.model, json_messages,
                temperature=self.temperature, response_format={"type": "json_object"},
            )
        except LLMError:
            return None
        value, numeric, reasoning = parse_json_verdict(raw, rubric)
        if numeric is None:  # valid call but not JSON-with-grade → try the GRADE regex on it
            value, numeric, reasoning = parse_verdict(raw, rubric)
        return JudgeOutput(value, numeric, reasoning, json_prompt, raw)
