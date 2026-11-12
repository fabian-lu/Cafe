"""Run a judge over a study's answers, with replication and progress."""

from __future__ import annotations

import asyncio
import hashlib
import json
import warnings
from typing import Any, Callable

from cafe.execution.progress import progress_bar
from cafe.execution.results import Results
from cafe.judging.judge import Judge
from cafe.judging.ratings import Rating, Ratings
from cafe.judging.rubric import Rubric

ProgressFn = Callable[[Rating, int, int], None]

#: The settings that, besides its ``model``, change what an LLM judge is asked (an
#: :class:`~cafe.LLMJudge` has them; a custom judge usually doesn't). Settings that change how
#: it is asked, not what (``temperature``, ``structured``, ``timeout``, ``retries``), are left
#: out: a resumed run may change them and keep its verdicts.
_JUDGE_PROMPT_SETTINGS = ("preset", "prompt_template", "system_prompt")


def _judged_by(judge: Judge, rubric: Rubric) -> dict[str, str]:
    """Who writes this run's verdicts, as a judge checkpoint records it with each verdict.

    ``judge`` and ``rubric`` are the names a person reads in the file and in warnings;
    ``digest`` fingerprints the judge (its ``model`` plus :data:`_JUDGE_PROMPT_SETTINGS`)
    together with the whole rubric (name, scale type, levels, instruction, prompt template),
    and is what a resumed run compares.
    """
    model = str(getattr(judge, "model", "judge"))
    identity = {
        "judge": [model, {name: getattr(judge, name, None) for name in _JUDGE_PROMPT_SETTINGS}],
        "rubric": [rubric.name, rubric.scale_type,
                   [[lvl.value, lvl.label, lvl.description] for lvl in rubric.levels],
                   rubric.instruction, rubric.prompt_template],
    }
    text = json.dumps(identity, sort_keys=True, default=str)
    return {"judge": model, "rubric": rubric.name,
            "digest": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]}


def _describe_sources(sources: list[Any], current: dict[str, str]) -> str:
    """Name the judges and rubrics that wrote dropped verdicts, and this run's, for a warning."""
    def names(src: Any) -> tuple[str, str]:
        src = src if isinstance(src, dict) else {}
        return str(src.get("judge", "?")), str(src.get("rubric", "?"))

    here = (current["judge"], current["rubric"])
    stored = sorted({names(src) for src in sources})
    shown = ", ".join(f"judge {j!r} on rubric {r!r}" for j, r in stored[:3])
    if len(stored) > 3:
        shown += f" and {len(stored) - 3} more"
    text = f"stored: {shown}; this run: judge {here[0]!r} on rubric {here[1]!r}"
    if here in stored:
        text += (" — same names, but another preset, prompt template or system prompt, or "
                 "another scale, levels, instruction or template")
    return text


async def judge_results(
    results: Results,
    judge: Judge,
    rubric: Rubric,
    *,
    repetitions: int = 1,
    concurrency: int = 8,
    references: dict[str, str] | None = None,
    questions: dict[str, str] | None = None,
    checkpoint_path: str | None = None,
    resume: bool = True,
    retry_errors: bool = True,
    on_progress: ProgressFn | None = None,
    progress: bool = False,
) -> Ratings:
    """Judge every successful answer ``repetitions`` times.

    ``questions`` / ``references`` map ``input_id`` -> text so the judge sees the
    original question and any gold answer.

    Parameters
    ----------
    checkpoint_path:
        Makes the (often expensive) judging phase crash-safe: each verdict is appended
        as it lands and a resumed run skips already-scored ``(answer, judge_rep)``
        pairs — the same guarantee :func:`cafe.run_study` gives the answer phase. Only
        verdicts for the answers judged now, with ``judge_rep < repetitions``, are
        carried forward; any other rows in the file are dropped with a warning.
    resume:
        Load ``checkpoint_path`` and skip what it already holds. ``False`` judges
        everything again (still appending to the file).
    retry_errors:
        On resume, re-judge checkpointed verdicts that failed (unparseable / judge
        error); the retry replaces them. ``False`` keeps them as done.

    Warns
    -----
    UserWarning
        If the checkpoint holds verdicts that don't belong to this run: for other answers
        or ``judge_rep >= repetitions`` (the answers or ``repetitions`` changed since it was
        written), written by another judge or rubric, or not on the rubric's scale. They
        are left out of the result, and this run's answers among them are judged again.
        Also if it holds verdicts that don't record who wrote them, which are kept (see
        Notes).

    Notes
    -----
    Each checkpointed verdict records who wrote it: the judge, by its ``model`` and, for an
    :class:`~cafe.LLMJudge`, its ``preset``, ``prompt_template`` and ``system_prompt``; and
    the rubric, by its name, scale type, levels, instruction and prompt template. A resumed
    run carries forward only the verdicts this judge wrote on this rubric, so reusing a
    checkpoint path with another judge or rubric (e.g. ``rejudge(other_judge,
    checkpoint_path=...)`` on the file of the first run) judges every answer again instead
    of returning the old verdicts under the new judge's name. A custom judge is known by
    its ``model`` alone, so give differently configured custom judges different ``model``
    names. A row from a file written before checkpoints recorded this is kept as this
    judge's verdict, with a warning, unless its verdict is not on the rubric's scale:
    every carried-forward verdict is checked with :meth:`~cafe.Rubric.numeric`.
    """
    questions = questions or {}
    references = references or {}
    # Resolve one-time judge setup (e.g. structured-output capability) once, before the
    # loop — so a capability probe runs at most once, never per judged answer.
    if hasattr(judge, "prepare"):
        await judge.prepare()

    # An empty-string answer is a valid (bad) answer — judge it (it will score low). Only a
    # genuinely absent output (None) is unjudgeable; record that as an explicit error rating
    # so it is visible in ratings.failures() rather than silently dropped.
    targets = [o for o in results.observations if o.ok and o.output is not None]
    unjudgeable = [o for o in results.observations if o.ok and o.output is None]

    # Resume from a ratings checkpoint if given. Each verdict is stored with who wrote it
    # (judged_by: this judge and rubric), so a resumed run can tell its own verdicts apart.
    ckpt = None
    judged_by = _judged_by(judge, rubric)
    prior: dict[str, tuple[Rating, Any]] = {}
    if checkpoint_path is not None:
        from cafe.execution.checkpoint import RatingsCheckpoint

        ckpt = RatingsCheckpoint(checkpoint_path, judged_by=judged_by)
        if resume:
            prior = ckpt.load_judged()

    def _key(obs_key: str, judge_rep: int) -> str:
        return f"{obs_key}::jr{judge_rep}"

    # Only carry forward checkpoint verdicts that belong to the CURRENT request (these
    # answers x these repetitions, by this judge on this rubric) — as run_study does for
    # answers. Verdicts for answers no longer judged, or for judge_reps beyond
    # ``repetitions``, would silently enter the statistics; so would another judge's or
    # rubric's, relabelled as this judge's verdicts on this rubric's scale. A row that
    # doesn't record who wrote it (a file from before checkpoints did) is kept, with a
    # warning; as a backstop, every carried-forward verdict must be on this rubric's scale.
    # retry_errors: a checkpointed verdict that failed (unparseable / judge error) is
    # re-judged rather than treated as done, and its retry replaces it (the checkpoint
    # stays append-only; load() is last-wins per key).
    valid = {_key(o.key(), jr) for o in targets for jr in range(repetitions)}
    kept: dict[str, Rating] = {}
    retried = unrecorded = other_answers = off_scale = 0
    other_sources: list[Any] = []
    for k, (r, source) in prior.items():
        if k not in valid:
            other_answers += 1
        elif source is not None and (not isinstance(source, dict)
                                     or source.get("digest") != judged_by["digest"]):
            other_sources.append(source)
        elif retry_errors and r.error is not None:
            retried += 1
        elif r.value_numeric is not None and rubric.numeric(r.value_numeric) is None:
            off_scale += 1
        else:
            kept[k] = r
            if source is None:
                unrecorded += 1
    if unrecorded:
        warnings.warn(
            f"ratings checkpoint {checkpoint_path!r}: {unrecorded} stored verdict(s) don't "
            "record which judge and rubric wrote them (the file predates CAFE recording that); "
            f"they are kept as this run's: judge {judged_by['judge']!r} on rubric "
            f"{rubric.name!r}. If another judge or rubric wrote them, use a new checkpoint "
            "path so the answers are judged again.",
            stacklevel=2,
        )
    stale = len(prior) - len(kept) - retried
    if stale:
        reasons = []
        if other_answers:
            reasons.append(f"{other_answers} for other answers, or judge_rep >= repetitions "
                           "(the study or repetitions changed since the checkpoint was written)")
        if other_sources:
            reasons.append(f"{len(other_sources)} written by another judge or rubric "
                           f"({_describe_sources(other_sources, judged_by)}), now judged again "
                           "— use one checkpoint file per judge and rubric")
        if off_scale:
            reasons.append(f"{off_scale} whose verdict is not on the scale of rubric "
                           f"{rubric.name!r}, now judged again")
        warnings.warn(
            f"ratings checkpoint {checkpoint_path!r}: dropped {stale} stored verdict(s) that "
            f"don't belong to this judging run: {'; '.join(reasons)}. Resumed ratings contain "
            "only this run's verdicts.",
            stacklevel=2,
        )

    todo = [(o, jr) for o in targets for jr in range(repetitions)
            if _key(o.key(), jr) not in kept]
    total = len(targets) * repetitions
    ratings: list[Rating] = list(kept.values())
    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    done = len(kept)

    with progress_bar(total, "judging", enabled=progress and on_progress is None) as bar:
        report = on_progress or bar

        async def one(obs, judge_rep: int) -> None:
            nonlocal done
            question = questions.get(obs.input_id, "")
            reference = references.get(obs.input_id)
            answer = obs.output if isinstance(obs.output, str) else str(obs.output)
            async with sem:
                out = await judge.score(rubric, question, answer, reference)
            rating = Rating(
                obs_key=obs.key(),
                config=dict(obs.config),
                input_id=obs.input_id,
                rep=obs.rep,
                judge_rep=judge_rep,
                value=out.value,
                value_numeric=out.value_numeric,
                reasoning=out.reasoning,
                error=None if out.value_numeric is not None else (out.reasoning or "unparseable"),
                prompt=out.prompt,
                raw_response=out.raw_response,
            )
            async with lock:
                ratings.append(rating)
                if ckpt is not None:
                    ckpt.append(rating)
                done += 1
                if report is not None:
                    report(rating, done, total)

        if todo:
            await asyncio.gather(*(one(o, jr) for o, jr in todo))

    # Explicit error rows for answers that produced no output at all.
    for obs in unjudgeable:
        ratings.append(Rating(
            obs_key=obs.key(), config=dict(obs.config), input_id=obs.input_id,
            rep=obs.rep, judge_rep=0, value=None, value_numeric=None,
            error="unjudgeable: the system produced no output (None)",
        ))

    return Ratings(
        rubric=rubric,
        judge_model=getattr(judge, "model", "judge"),
        factors=list(results.factors),
        items=ratings,
        judge_system_prompt=getattr(judge, "system_prompt", None),
    )
