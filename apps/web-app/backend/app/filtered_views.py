"""Question-filtered results views, computed once per stored result and kept in the database.

``GET /api/studies/{id}/results?fkey=…&fval=…`` refits the whole statistics stack on the questions
whose metadata ``fkey`` equals ``fval`` (:func:`app.runner.filtered_payload`): a few seconds up to
about a minute per view. So a computed view is kept in two layers, both keyed by the StudyResult
row it was computed from:

1. an in-memory dict, the first layer (this process only, at most ``_MEMORY_MAX`` views);
2. the ``filtered_results`` table (:class:`app.models.FilteredResult`), so a view outlives a
   backend restart.

Every view is kept with the :func:`view_version` of the payload it was computed from, and a lookup
only returns a view whose version matches the StudyResult's payload *now*: after a re-run, a
statistics refresh or a statistics-version bump the view is computed again, never served stale.
Whatever replaces or deletes a StudyResult also calls :func:`forget`, so the views of results that
are gone don't stay in the table.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import traceback
from collections.abc import Iterable

from sqlalchemy import delete, select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncSession

from app import models, runner
from app.db import SessionLocal

# The first layer: (study_id, study_result_id, fkey, fval) -> (version, view payload)
_MEMORY: dict[tuple[int, int, str, str], tuple[str, dict]] = {}
_MEMORY_MAX = 64

# Views being computed right now, by (study_result_id, fkey, fval, version): concurrent requests
# for the same view wait for one computation instead of each refitting the models.
_IN_FLIGHT: dict[tuple[int, str, str, str], asyncio.Task] = {}

# INSERT … ON CONFLICT for the databases the app runs on (see app/config.py)
_INSERT = {"postgresql": postgresql.insert, "sqlite": sqlite.insert}


def view_version(payload: dict) -> str:
    """Name the computation a stored results payload comes from.

    Parameters
    ----------
    payload : dict
        A ``StudyResult.payload``.

    Returns
    -------
    str
        The statistics version of this backend (:data:`app.runner.STATS_VERSION`) and the
        payload's ``computed_at``, which every run, import and statistics refresh sets anew. A
        payload stored before payloads carried ``computed_at`` is named by a digest of its
        content instead.
    """
    computed_at = payload.get("computed_at")
    if computed_at is None:
        content = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        source = "sha256=" + hashlib.sha256(content).hexdigest()
    else:
        source = f"computed_at={computed_at!r}"
    return f"stats_version={runner.STATS_VERSION};{source}"


async def lookup(db: AsyncSession, study_id: int, result_id: int, fkey: str, fval: str,
                 version: str) -> dict | None:
    """Return a view of the current results that was computed before: from memory, else from
    the database.

    Parameters
    ----------
    db : AsyncSession
        The request's session.
    study_id : int
        The study.
    result_id : int
        The StudyResult row the view filters.
    fkey, fval : str
        The question-metadata key and the value it must equal.
    version : str
        :func:`view_version` of that row's payload now.

    Returns
    -------
    dict or None
        The view, or ``None`` when neither layer holds one computed from this payload.
    """
    key = (study_id, result_id, fkey, fval)
    hit = _MEMORY.get(key)
    if hit is not None and hit[0] == version:
        return hit[1]
    view = (await db.execute(select(models.FilteredResult.payload).where(
        models.FilteredResult.study_result_id == result_id,
        models.FilteredResult.fkey == fkey,
        models.FilteredResult.fval == fval,
        models.FilteredResult.version == version))).scalar_one_or_none()
    if view is not None:
        _remember(key, version, view)
    return view


async def compute(study_id: int, result_id: int, payload: dict, items: list[dict], fkey: str,
                  fval: str, version: str) -> dict:
    """Compute a view, keep it in both layers, and return it.

    The models are refitted in a worker thread (:func:`app.runner.filtered_payload`), so the app
    stays responsive. Concurrent requests for the same view share one computation, which runs as
    a task of its own and stores the view with its own session: the view is kept even when the
    request that started it is gone. Only views of a value ``GET /filters`` offers are stored in
    the database, so arbitrary query strings can't fill the table.

    Parameters
    ----------
    study_id : int
        The study.
    result_id : int
        The StudyResult row whose payload is filtered.
    payload : dict
        That row's payload.
    items : list of dict
        The study's dataset items, which carry the question metadata.
    fkey, fval : str
        The question-metadata key and the value it must equal.
    version : str
        :func:`view_version` of ``payload``.

    Returns
    -------
    dict
        The filtered results payload.

    Raises
    ------
    Exception
        Whatever the recompute raises. Storing never fails the request: a view that can't be
        stored is returned all the same.
    """
    key = (result_id, fkey, fval, version)
    task = _IN_FLIGHT.get(key)
    if task is None:
        task = asyncio.create_task(
            _compute_and_store(study_id, result_id, payload, items, fkey, fval, version))
        _IN_FLIGHT[key] = task
        task.add_done_callback(lambda _task: _IN_FLIGHT.pop(key, None))
    return await asyncio.shield(task)


async def _compute_and_store(study_id: int, result_id: int, payload: dict, items: list[dict],
                             fkey: str, fval: str, version: str) -> dict:
    """The task behind :func:`compute` (same parameters and return value): refit in a worker
    thread, keep the view in memory, and store it in the database if its value is offered."""
    view = await asyncio.to_thread(runner.filtered_payload, payload, items, fkey, fval)
    _remember((study_id, result_id, fkey, fval), version, view)
    if fval in runner.dataset_filters(items).get(fkey, []):
        try:
            async with SessionLocal() as db:
                if await store(db, result_id, fkey, fval, version, view):
                    await db.commit()
        except Exception:  # noqa: BLE001 — the table only saves recomputing: serve the view anyway
            traceback.print_exc()
    return view


async def store(db: AsyncSession, result_id: int, fkey: str, fval: str, version: str,
                view: dict) -> bool:
    """Insert one view into the ``filtered_results`` table, or replace the one stored for an
    older payload. Doesn't commit.

    A single upsert on the view's unique key, so two requests (or backend processes) storing the
    same view at once don't conflict: the last one wins.

    Parameters
    ----------
    db : AsyncSession
        The session to write with.
    result_id : int
        The StudyResult row the view filters.
    fkey, fval : str
        The question-metadata key and the value it must equal.
    version : str
        :func:`view_version` of the payload the view was computed from.
    view : dict
        The filtered results payload.

    Returns
    -------
    bool
        ``False`` (and nothing is stored) if the StudyResult no longer exists, e.g. because the
        study was deleted while the view was computed.
    """
    exists = select(models.StudyResult.id).where(models.StudyResult.id == result_id)
    if (await db.execute(exists)).first() is None:
        return False
    insert = _INSERT[db.get_bind().dialect.name](models.FilteredResult).values(
        study_result_id=result_id, fkey=fkey, fval=fval, version=version, payload=view)
    await db.execute(insert.on_conflict_do_update(
        index_elements=["study_result_id", "fkey", "fval"],
        set_={"version": insert.excluded.version, "payload": insert.excluded.payload,
              "computed_at": insert.excluded.computed_at}))
    return True


async def forget(db: AsyncSession, study_id: int, result_ids: Iterable[int] | None = None) -> None:
    """Drop the stored views of a study's results: all of them, or those of ``result_ids``.

    Call it in the transaction that replaces or deletes those results (a re-run, a statistics
    refresh, deleting the study), so the views go with them. Doesn't commit.

    Parameters
    ----------
    db : AsyncSession
        The session that replaces or deletes the results.
    study_id : int
        The study.
    result_ids : iterable of int, optional
        Only the views of these StudyResult rows. Default: those of all the study's rows (call
        it before deleting the rows, then).
    """
    rows = delete(models.FilteredResult).execution_options(synchronize_session=False)
    if result_ids is None:
        rows = rows.where(models.FilteredResult.study_result_id.in_(
            select(models.StudyResult.id).where(models.StudyResult.study_id == study_id)))
    else:
        result_ids = list(result_ids)
        rows = rows.where(models.FilteredResult.study_result_id.in_(result_ids))
    await db.execute(rows)
    for key in [k for k in _MEMORY
                if k[0] == study_id and (result_ids is None or k[1] in result_ids)]:
        del _MEMORY[key]


def _remember(key: tuple[int, int, str, str], version: str, view: dict) -> None:
    """Keep a view in the first layer, dropping the oldest one when it is full."""
    _MEMORY.pop(key, None)
    if len(_MEMORY) >= _MEMORY_MAX:
        del _MEMORY[next(iter(_MEMORY))]
    _MEMORY[key] = (version, view)
