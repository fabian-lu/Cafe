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
    on_progress: ProgressFn | None = None,
    progress: bool = False,
) -> Ratings:
    """Judge every successful answer ``repetitions`` times.

    ``questions`` / ``references`` map ``input_id`` -> text so the judge sees the
    original question and any gold answer.

    ``checkpoint_path`` makes the (often expensive) judging phase crash-safe: each verdict
    is appended as it lands and a resumed run skips already-scored ``(answer, judge_rep)``
    pairs — the same guarantee :func:`cafe.run_study` gives the answer phase.
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

    todo = [(o, jr) for o in targets for jr in range(repetitions)
            if _key(o.key(), jr) not in prior]
    total = len(targets) * repetitions
    ratings: list[Rating] = list(prior.values())
    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    done = len(prior)

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
