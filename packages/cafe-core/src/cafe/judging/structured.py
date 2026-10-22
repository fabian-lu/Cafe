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
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from cafe.judging.rubric import Rubric
from cafe.llm import LLMError, complete

#: Appended to the user prompt in structured mode. ``{grade}`` is the scale-aware hint
#: (a range for numeric, the exact allowed values for ordinal/binary).
JSON_INSTRUCTION = (
    '\n\nReturn ONLY a JSON object with keys "reasoning" (a brief string) and "grade" '
    "({grade}). Output nothing else."
)

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
