"""CAFE web platform — FastAPI backend.

A thin JSON API over cafe-core: it discovers the user's pipeline, manages studies/datasets/rubrics/
raters, runs studies as background tasks, and serves results as JSON for the frontend to draw. No
statistics or execution logic lives here — cafe-core does all of that.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import FRONTEND_ORIGIN, missing_llm_keys
from app.db import SessionLocal, init_db
from app.routers import crud, judge, pipeline, raters, runs
from app.seed import seed_if_empty


async def _fail_orphaned_runs() -> None:
    """Runs execute as in-process tasks and progress lives in memory, so any study
    still marked "running" at boot was orphaned by a crash/restart mid-run. Nothing
    else ever repairs the row — POST /run would 409 "already running" forever — so
    mark them failed here; the user can simply re-run them."""
    from sqlalchemy import update

    from app import models

    async with SessionLocal() as db:
        result = await db.execute(
            update(models.Study).where(models.Study.status == "running").values(status="failed")
        )
        await db.commit()
        if result.rowcount:
            print(f"[startup] marked {result.rowcount} orphaned 'running' study(ies) as failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()             # create tables if missing
    await _fail_orphaned_runs() # repair studies orphaned by a mid-run crash/restart
    await seed_if_empty()       # first-boot only: load the bundled demo study into an empty DB
    yield


app = FastAPI(title="CAFE Platform", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router)
app.include_router(crud.router)
app.include_router(runs.router)
app.include_router(raters.router)
app.include_router(judge.router)


@app.get("/api/health")
def health():
    """Liveness + a heads-up if LLM credentials aren't set (studies will fail without them)."""
    return {"status": "ok", "missing_llm_keys": missing_llm_keys()}
