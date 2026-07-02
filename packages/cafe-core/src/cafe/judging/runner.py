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
    on_progress: ProgressFn | None = None,
    progress: bool = False,
) -> Ratings:
    """Judge every successful answer ``repetitions`` times.

    ``questions`` / ``references`` map ``input_id`` -> text so the judge sees the
    original question and any gold answer.
    """
    questions = questions or {}
    references = references or {}
    # Resolve one-time judge setup (e.g. structured-output capability) once, before the
    # loop — so a capability probe runs at most once, never per judged answer.
    if hasattr(judge, "prepare"):
        await judge.prepare()
    targets = [o for o in results.observations if o.ok and o.output]
    total = len(targets) * repetitions
    ratings: list[Rating] = []
    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    done = 0

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
                done += 1
                if report is not None:
                    report(rating, done, total)

        await asyncio.gather(*(one(o, jr) for o in targets for jr in range(repetitions)))

    return Ratings(
        rubric=rubric,
        judge_model=getattr(judge, "model", "judge"),
        factors=list(results.factors),
        items=ratings,
        judge_system_prompt=getattr(judge, "system_prompt", None),
    )
