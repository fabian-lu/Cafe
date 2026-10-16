"""Run a judge over a study's answers, with replication and progress."""

from __future__ import annotations

import asyncio
from typing import Callable

from cafe.execution.progress import progress_bar
from cafe.execution.results import Results
from cafe.judging.judge import Judge
from cafe.judging.ratings import Rating, Ratings
from cafe.judging.rubric import Rubric

ProgressFn = Callable[[Rating, int, int], None]


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
        If the checkpoint holds verdicts that don't belong to this run (the answers or
        ``repetitions`` changed since it was written); they are left out of the result.
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

    # Resume from a ratings checkpoint if given.
    ckpt = None
    prior: dict[str, Rating] = {}
    if checkpoint_path is not None:
        from cafe.execution.checkpoint import RatingsCheckpoint

        ckpt = RatingsCheckpoint(checkpoint_path)
        if resume:
            prior = ckpt.load()

    def _key(obs_key: str, judge_rep: int) -> str:
        return f"{obs_key}::jr{judge_rep}"

    # Only carry forward checkpoint verdicts that belong to the CURRENT request (these
    # answers x these repetitions) — as run_study does for answers. Verdicts for answers no
    # longer judged, or for judge_reps beyond ``repetitions``, would silently enter the
    # statistics. retry_errors: a checkpointed verdict that failed (unparseable / judge
    # error) is re-judged rather than treated as done, and its retry replaces it (the
    # checkpoint stays append-only; load() is last-wins per key).
    valid = {_key(o.key(), jr) for o in targets for jr in range(repetitions)}
    kept = {
        k: r for k, r in prior.items()
        if k in valid and not (retry_errors and r.error is not None)
    }
    retried = sum(
        1 for k, r in prior.items() if k in valid and retry_errors and r.error is not None
    )
    stale = len(prior) - len(kept) - retried
    if stale:
        import warnings

        warnings.warn(
            f"ratings checkpoint {checkpoint_path!r}: dropped {stale} stored verdict(s) that "
            "don't belong to this judging run (other answers, or judge_rep >= repetitions — "
            "the study or repetitions changed since the checkpoint was written); resumed "
            "ratings contain only the current answers' verdicts.",
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
