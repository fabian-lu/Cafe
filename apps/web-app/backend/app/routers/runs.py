"""Launch a study run, stream its progress (SSE), and serve cached results."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app import models, runner
from app.db import SessionLocal, get_session

# filtered-results cache: (study_id, result_row_id, dimension, fkey, fval) -> payload
_FILTER_CACHE: dict[tuple, dict] = {}

# /stream: how long a DB-"running" study may lack an in-memory progress entry before the
# stream reports that no run is active in this process
_NO_TASK_GRACE_S = 2.0

router = APIRouter(prefix="/api", tags=["runs"])


@router.post("/studies/{id_}/run")
async def run_study(id_: int, db: AsyncSession = Depends(get_session)):
    study = await db.get(models.Study, id_)
    if study is None:
        raise HTTPException(404, "study not found")
    # Claim the study with ONE conditional UPDATE, so the check and the write are atomic in the
    # DB: a separate read-check-write let two requests arriving together (double-click, two
    # tabs) both see "not running" and both launch a full run.
    claimed = await db.execute(
        update(models.Study)
        .where(models.Study.id == id_, models.Study.status != "running")
        .values(status="running", progress=0))
    await db.commit()
    if claimed.rowcount == 0:
        raise HTTPException(409, "study is already running")
    runner.launch(id_)
    return {"status": "running"}


@router.post("/studies/{id_}/estimate")
async def estimate(id_: int, db: AsyncSession = Depends(get_session)):
    """Preflight → time/cost estimate for the full run. Slow (runs one input per config, real calls),
    so the UI shows a spinner while it computes."""
    study = await db.get(models.Study, id_)
    if study is None:
        raise HTTPException(404, "study not found")
    try:
        return await runner.estimate_study(id_)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"estimate failed: {exc}")


@router.get("/studies/{id_}/stream")
async def stream_progress(id_: int):
    """Server-Sent Events: emit the study's progress until it finishes.

    A study with no run in this process gets a single event and the stream ends, instead of an
    idle stream that never terminates: its DB status (a draft, or a run that finished before a
    restart), or ``"failed"`` for a "running" row whose run task is gone. 404 if no such study."""
    async with SessionLocal() as db:  # short-lived: don't hold a DB connection for the stream
        study = await db.get(models.Study, id_)
        if study is None:
            raise HTTPException(404, "study not found")
        db_status = study.status

    async def gen():
        last = None
        waited = 0.0
        while True:
            p = runner.PROGRESS.get(id_)
            if p is None:
                if db_status == "running" and waited < _NO_TASK_GRACE_S:
                    # the run task registers itself as soon as it is scheduled; give it a moment
                    waited += 0.5
                    await asyncio.sleep(0.5)
                    continue
                if db_status == "running":
                    p = {"phase": "error", "status": "failed",
                         "error": "no active run for this study in this server process "
                                  "(interrupted by a backend restart?)"}
                else:
                    p = {"phase": "idle", "status": db_status}
                yield f"data: {json.dumps(p)}\n\n"
                break
            snapshot = json.dumps(p)
            if snapshot != last:
                yield f"data: {snapshot}\n\n"
                last = snapshot
            if p.get("status") in ("done", "failed"):
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/studies/{id_}/dimensions")
async def list_dimensions(id_: int, db: AsyncSession = Depends(get_session)):
    """The judged dimensions (rubric names) this study has results for, in creation order.
    The first entry is the primary dimension (what /results returns without a param)."""
    if await db.get(models.Study, id_) is None:
        raise HTTPException(404, "study not found")
    rows = list((await db.execute(
        select(models.StudyResult).where(models.StudyResult.study_id == id_)
        .order_by(models.StudyResult.id)
    )).scalars())
    return [{"dimension": r.dimension or "quality"} for r in rows]


@router.get("/studies/{id_}/filters")
async def get_filters(id_: int, db: AsyncSession = Depends(get_session)):
    """Filterable question-metadata keys/values of this study's dataset (non-reserved item
    keys with 2-30 distinct values). Empty dict = nothing to filter by."""
    study = await db.get(models.Study, id_)
    if study is None:
        raise HTTPException(404, "study not found")
    dataset = await db.get(models.Dataset, study.dataset_id) if study.dataset_id else None
    return runner.dataset_filters(list(dataset.items) if dataset else [])


@router.get("/studies/{id_}/results")
async def get_results(id_: int, dimension: str | None = None,
                      fkey: str | None = None, fval: str | None = None,
                      db: AsyncSession = Depends(get_session)):
    q = select(models.StudyResult).where(models.StudyResult.study_id == id_)
    if dimension is not None:
        q = q.where(models.StudyResult.dimension == dimension)
    res = (await db.execute(q.order_by(models.StudyResult.id))).scalars().first()
    if res is None:
        raise HTTPException(404, "no results yet — run the study first"
                            if dimension is None else f"no results for dimension {dimension!r}")
    study = await db.get(models.Study, id_)
    dataset = await db.get(models.Dataset, study.dataset_id) if study and study.dataset_id else None
    items = list(dataset.items) if dataset else []

    # Stored by an older statistics version (a run/import before an update, the bundled demo
    # seed): recompute once from the stored records — no LLM calls — and keep the result. Only
    # judged results have statistics to recompute; an unjudged study's payload is left alone.
    payload, res_id, res_dimension = res.payload, res.id, res.dimension  # read before any rollback
    if payload.get("stats_version") != runner.STATS_VERSION and payload.get("rubric"):
        try:
            fresh = await asyncio.to_thread(runner.refresh_payload, payload, items)
            res.payload = fresh
            await db.commit()
            payload = fresh
        except Exception:  # noqa: BLE001 — serve what's stored rather than fail the page
            await db.rollback()  # (`payload` is the copy read before; `res` is expired now)
    if not fkey or fval is None:
        return payload

    # question-subset view: recompute the whole stats stack on the filtered questions
    # (keyed by when the stored payload was computed, so a re-run doesn't serve stale views)
    cache_key = (id_, res_id, res_dimension, payload.get("computed_at"), fkey, fval)
    if cache_key in _FILTER_CACHE:
        return _FILTER_CACHE[cache_key]
    if not items:
        raise HTTPException(400, "this study's dataset carries no question metadata to filter by")
    try:
        # refits every model on the subset (R included): in a worker thread, so the app stays responsive
        payload = await asyncio.to_thread(runner.filtered_payload, payload, items, fkey, fval)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"filtered recompute failed: {exc}")
    if len(_FILTER_CACHE) > 64:
        _FILTER_CACHE.clear()
    _FILTER_CACHE[cache_key] = payload
    return payload
