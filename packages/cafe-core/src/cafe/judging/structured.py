"""Structured (JSON) verdicts — ask the judge for JSON instead of a ``GRADE:`` line.

Regex-parsing free text is the usual source of unparseable (NaN) verdicts, especially
with reasoning models that bury the grade in a think block or return empty prose. When
a model supports it, we instead request a JSON verdict ``{"reasoning", "grade"}`` via
LiteLLM's ``response_format`` — far more reliable.

Capability is decided **per model**, robustly:
  1. trust LiteLLM's static map (``supports_response_schema``);
  2. if that says no/unknown, **probe once** (cached) — some endpoints honour
     ``response_format`` without being in the map (e.g. Ollama-served models like gpt-oss);
  3. if the probe also fails, the judge uses the ``GRADE:`` parser instead.

If the provider rejects ``response_format`` the judge makes a plain call instead, and a
reply that isn't usable JSON with a ``grade`` goes to the regex parser, so structured mode
can never do worse than the default. A JSON verdict whose grade is off the rubric's scale
is an error rating: the regex is not re-run over the JSON text, where it could pick up a
``grade: N`` from inside the reasoning.

Only a refused request falls back to the plain call: HTTP 400, 404 or 422, the way
providers, and LiteLLM's own check, reject a ``response_format``. A timeout, a rate limit,
or a server, connection or auth error would fail the plain call the same way, after the
same wait and on the same quota, so it is an error rating at once, as when a plain call
fails.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from cafe.judging.prompts import JUDGE_PRESETS
from cafe.judging.rubric import Rubric
from cafe.llm import LLMError, complete

#: Replaces a preset's closing ``GRADE:`` instruction in structured mode. ``{grade}`` is the
#: scale-aware hint (a range for numeric, the exact allowed values for ordinal/binary).
JSON_INSTRUCTION = (
    '\n\nReturn ONLY a JSON object with keys "reasoning" (a brief string) and "grade" '
    "({grade}). Output nothing else."
)
#: Appended after a custom prompt template instead, whose own ``GRADE:`` request is kept.
JSON_INSTEAD_INSTRUCTION = (
    "\n\nInstead of ending with a GRADE: line, return ONLY a JSON object with keys "
    '"reasoning" (a brief string) and "grade" ({grade}). Output nothing else.'
)


def build_json_prompt(prompt: str, rubric: Rubric) -> str:
    """The structured-mode user prompt, built from the plain-mode one.

    Parameters
    ----------
    prompt:
        The rendered plain-mode prompt (:func:`~cafe.judging.prompts.build_judge_prompt`).
    rubric:
        The rubric being judged; supplies the scale-aware grade hint.

    Returns
    -------
    str
        For a built-in preset, ``prompt`` with its closing "finish with ``GRADE: <…>``"
        paragraph replaced by :data:`JSON_INSTRUCTION`, so the judge is not asked for a
        ``GRADE:`` line *and* for JSON only. A custom template's wording can't be cut
        reliably, so it is kept and followed by :data:`JSON_INSTEAD_INSTRUCTION`.
    """
    grade = rubric.grade_hint()
    for template in JUDGE_PRESETS.values():
        closing = "\n\n" + template.rsplit("\n\n", 1)[-1].replace("{grade}", grade)
        if prompt.endswith(closing):
            return prompt[: -len(closing)] + JSON_INSTRUCTION.format(grade=grade)
    return prompt + JSON_INSTEAD_INSTRUCTION.format(grade=grade)


# model string -> does it support response_format JSON (resolved once, then cached)
_support_cache: dict[str, bool] = {}
_probe_lock = asyncio.Lock()


async def supports_structured(model: str) -> bool:
    """Whether ``model`` can return JSON via ``response_format``.

    Static capability map first; on a no/unknown, a single cached runtime probe.
    """
    if model in _support_cache:
        return _support_cache[model]
    try:
        import litellm

        if litellm.supports_response_schema(model=model):
            _support_cache[model] = True
            return True
    except Exception:  # noqa: BLE001 — map lookup is best-effort
        pass

    async with _probe_lock:
        if model in _support_cache:  # another coroutine probed while we waited
            return _support_cache[model]
        ok = await _probe(model)
        _support_cache[model] = ok
        return ok


async def _probe(model: str) -> bool:
    """One cheap call: does ``model`` actually return valid JSON under response_format?"""
    try:
        out = await complete(
            model,
            [{"role": "user", "content": 'Reply with exactly this JSON: {"ok": 1}'}],
            response_format={"type": "json_object"},
            timeout=30,
        )
        return isinstance(_loads(out), dict)
    except (LLMError, ValueError, json.JSONDecodeError):
        return False


#: HTTP statuses with which a provider refuses a request for what it asks, as it answers a
#: ``response_format`` it doesn't support: 400 (LiteLLM's own ``UnsupportedParamsError`` is a
#: 400 too), 422 (e.g. Text Generation Inference) or 404 (OpenRouter, when no endpoint can
#: handle the requested parameters).
_REFUSED_STATUSES = frozenset({400, 404, 422})


def _refused_request(exc: LLMError) -> bool:
    """Whether a failed structured call was refused for what it asked, perhaps for its
    ``response_format``: the one failure the plain call can get past.

    :func:`cafe.complete` raises every provider error as an :class:`~cafe.llm.LLMError` whose
    ``__cause__`` is LiteLLM's exception, which carries the HTTP status. A refusal comes back
    at once, so a plain call after one costs a quick request, even when the refusal was for
    something else (a prompt too long for the model fails the plain call too). A timeout
    (408, 504), a rate limit (429), a server or connection error (5xx), an auth, payment or
    permission error (401–403), or an error raised before any request (no cause) has nothing
    to do with ``response_format``: the plain call would fail the same way, after the same
    wait and retries and on the same quota.

    Parameters
    ----------
    exc:
        The error the structured call raised.

    Returns
    -------
    bool
        ``True`` if its cause reports one of the statuses in :data:`_REFUSED_STATUSES`.
    """
    return getattr(exc.__cause__, "status_code", None) in _REFUSED_STATUSES


def _loads(raw: str) -> Any:
    """Parse JSON, tolerating code fences / surrounding prose around one object.

    If ``raw`` isn't JSON as a whole, scan it for complete top-level ``{...}`` objects
    (decoded with the JSON parser, so braces in surrounding prose such as ``{grade}`` are
    skipped). Exactly one must be found; two candidate objects are ambiguous and stay a
    parse error rather than a guess.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        decoder, found = json.JSONDecoder(), []
        start = raw.find("{")
        while start != -1:
            try:
                obj, end = decoder.raw_decode(raw, start)
            except json.JSONDecodeError:
                start = raw.find("{", start + 1)
                continue
            found.append(obj)
            start = raw.find("{", end)
        if len(found) == 1:
            return found[0]
        raise exc


def parse_json_verdict(raw: str, rubric: Rubric) -> tuple[Any, int | None, str | None]:
    """Extract ``(value, numeric, reasoning)`` from a JSON verdict.

    Returns
    -------
    tuple
        ``(None, None, None)`` if ``raw`` isn't usable JSON with a ``grade`` — the only
        signal for the caller to fall back to the ``GRADE:`` regex parser. If the grade
        is present but not on the rubric's scale, ``(grade, None, <error message>)`` —
        an error verdict, as from :func:`cafe.judging.prompts.parse_verdict`. Otherwise
        ``(grade, scale point, reasoning)``.
    """
    try:
        data = _loads(raw)
        grade = data["grade"]
    except (json.JSONDecodeError, TypeError, KeyError):
        return None, None, None
    numeric = rubric.numeric(grade)
    if numeric is None:
        return grade, None, f"verdict {grade!r} not on scale {rubric.min_value}-{rubric.max_value}"
    reasoning = data.get("reasoning") if isinstance(data, dict) else None
    return grade, numeric, reasoning
