"""Launch a study run, stream its progress (SSE), and serve cached results."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import models, runner
from app.db import get_session

router = APIRouter(prefix="/api", tags=["runs"])


@router.post("/studies/{id_}/run")
async def run_study(id_: int, db: AsyncSession = Depends(get_session)):
    study = await db.get(models.Study, id_)
    if study is None:
        raise HTTPException(404, "study not found")
    if study.status == "running":
        raise HTTPException(409, "study is already running")
    study.status = "running"
    study.progress = 0
    await db.commit()
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
    """Server-Sent Events: emit the study's progress until it finishes."""
    async def gen():
        last = None
        while True:
            p = runner.PROGRESS.get(id_, {"phase": "idle", "status": "idle"})
            snapshot = json.dumps(p)
            if snapshot != last:
                yield f"data: {snapshot}\n\n"
                last = snapshot
            if p.get("status") in ("done", "failed"):
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/studies/{id_}/results")
async def get_results(id_: int, db: AsyncSession = Depends(get_session)):
    res = (await db.execute(
        select(models.StudyResult).where(models.StudyResult.study_id == id_)
    )).scalar_one_or_none()
    if res is None:
        raise HTTPException(404, "no results yet — run the study first")
    return res.payload
