"""The execution engine: expand a design and run every cell, robustly.

Design goals (carried from the DIVA worker, made library-mode-friendly):

- **Incremental + resumable.** Each observation is written to the checkpoint the
  moment it completes; a resumed run skips what's already done.
- **Per-item error isolation.** A failing cell is recorded with its error and does
  not abort the rest of the study.
- **Configurable async concurrency.** Cells run concurrently up to a semaphore
  bound; tune to your rate limits.

This is the same engine the web platform's worker will wrap — there is no logic
here that depends on a database or a web server.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Callable

from cafe import design
from cafe.results import Observation, Results, config_label
from cafe.study import Study
from cafe.system import System, as_system, normalize_output

ProgressFn = Callable[[Observation, int, int], None]


def _input_id(item: Any, index: int) -> str:
    """Stable id for an input item: its ``id`` field if present, else position."""
    if isinstance(item, dict) and "id" in item:
        return str(item["id"])
    return f"in{index}"


async def _run_cell(
    system: System, config: dict[str, Any], item: Any, input_id: str, rep: int
) -> Observation:
    """Execute one cell. Never raises — failures are captured on the Observation."""
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    try:
        raw = await system.run(dict(config), item)
        output, metadata = normalize_output(raw)
        error: str | None = None
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — isolation is the whole point
        output, metadata, error = None, {}, f"{type(exc).__name__}: {exc}"
    elapsed = round(time.monotonic() - t0, 6)
    return Observation(
        config=dict(config),
        input_id=input_id,
        rep=rep,
        output=output,
        error=error,
        elapsed_s=elapsed,
        started_at=started,
        metadata=metadata,
    )


async def run_study(
    study: Study,
    *,
    replications: int | None = None,
    concurrency: int = 8,
    checkpoint_path: str | None = None,
    resume: bool = True,
    smoke: bool = False,
    on_progress: ProgressFn | None = None,
) -> Results:
    """Run a study end-to-end and return its :class:`Results`.

    Parameters
    ----------
    replications:
        Repetitions per (config, input). Defaults to ``study.replications``.
        Replication is how CAFE measures run-to-run nondeterminism.
    concurrency:
        Max cells in flight at once.
    checkpoint_path:
        If given, observations are appended here as they complete and a resumed
        run skips already-done cells. If ``None``, the run is in-memory only.
    smoke:
        Preflight: one input through every config, a single replication, no
        judging — to confirm configs execute and to estimate cost before
        committing to the full study.
    on_progress:
        Called as ``on_progress(observation, completed, total)`` after each cell.
    """
    reps = study.replications if replications is None else replications
    if reps < 1:
        raise ValueError("replications must be >= 1")

    configs = design.generate(study)
    inputs = list(study.inputs)
    if smoke:
        inputs = inputs[:1]
        reps = 1
    if not inputs:
        raise ValueError("study has no inputs to run")

    system = as_system(study.system)

    # Resume: load prior observations and skip their keys.
    checkpoint = None
    done: dict[str, Observation] = {}
    if checkpoint_path is not None:
        from cafe.checkpoint import Checkpoint

        checkpoint = Checkpoint(checkpoint_path)
        if resume:
            done = checkpoint.load()

    # Build the full cell list, then drop the ones already done.
    pending: list[tuple[dict[str, Any], Any, str, int]] = []
    for cfg in configs:
        for idx, item in enumerate(inputs):
            in_id = _input_id(item, idx)
            for rep in range(reps):
                probe = Observation(config=cfg, input_id=in_id, rep=rep)
                if probe.key() not in done:
                    pending.append((cfg, item, in_id, rep))

    total = len(configs) * len(inputs) * reps
    observations: list[Observation] = list(done.values())

    sem = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    completed = len(done)

    async def worker(cfg: dict[str, Any], item: Any, in_id: str, rep: int) -> None:
        nonlocal completed
        async with sem:
            obs = await _run_cell(system, cfg, item, in_id, rep)
        async with write_lock:
            observations.append(obs)
            if checkpoint is not None:
                checkpoint.append(obs)
            completed += 1
            if on_progress is not None:
                on_progress(obs, completed, total)

    if pending:
        await asyncio.gather(*(worker(*cell) for cell in pending))

    return Results(
        study_name=study.name,
        factors=[f.name for f in study.factors],
        observations=observations,
    )


def estimate(results: Results, total_cells: int) -> dict[str, Any]:
    """Rough full-study time/cost estimate from a smoke/partial run.

    Uses mean per-cell time and any ``cost_usd`` metadata observed so far,
    extrapolated to ``total_cells``. Conservative and meant for the preflight.
    """
    oks = [o for o in results.observations if o.ok]
    times = [o.elapsed_s for o in oks if o.elapsed_s is not None]
    costs = [o.metadata.get("cost_usd") for o in oks if "cost_usd" in o.metadata]
    mean_t = sum(times) / len(times) if times else None
    mean_c = sum(costs) / len(costs) if costs else None
    return {
        "sampled_cells": len(oks),
        "total_cells": total_cells,
        "est_total_compute_s": round(mean_t * total_cells, 2) if mean_t else None,
        "est_total_cost_usd": round(mean_c * total_cells, 4) if mean_c else None,
        "labels": [config_label(o.config) for o in oks[:1]],
    }
