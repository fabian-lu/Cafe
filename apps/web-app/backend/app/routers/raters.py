"""Human rating + inter-rater reliability (Krippendorff's α).

A study's answers (from the cached results) are sampled into a rating sheet; humans score them by
name (no accounts). We align each human score with the judge's verdict by the answer's index in the
results record list, and compute α with cafe-core.
"""

from __future__ import annotations

import random

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import cafe

from app import models
from app.db import get_session

router = APIRouter(prefix="/api", tags=["raters"])

_METRIC = {"ordinal": "ordinal", "numeric": "interval", "binary": "nominal"}


async def _records(db: AsyncSession, study_id: int) -> list[dict]:
    res = (await db.execute(select(models.StudyResult).where(
        models.StudyResult.study_id == study_id))).scalar_one_or_none()
    if res is None:
        raise HTTPException(404, "no results yet — run the study first")
    return res.payload.get("records", [])


_DEFAULT_LEVELS = [{"value": i, "label": str(i), "description": ""} for i in range(6)]


async def _study_rubric(db: AsyncSession, study_id: int) -> dict:
    """The rubric the judge used, so humans rate on the SAME scale (levels/instruction)."""
    study = await db.get(models.Study, study_id)
    if study and study.rubric_id:
        r = await db.get(models.Rubric, study.rubric_id)
        if r:
            return {"name": r.name, "scale_type": r.scale_type, "instruction": r.instruction,
                    "levels": r.levels or _DEFAULT_LEVELS}
    return {"name": "score", "scale_type": "numeric", "instruction": "", "levels": _DEFAULT_LEVELS}


@router.get("/studies/{id_}/rating-sheet")
async def rating_sheet(id_: int, n: int = 40, db: AsyncSession = Depends(get_session)):
    """A stratified sample of answers to hand-rate — spans the judge's verdict range so α is
    meaningful — plus the study's rubric + judge model, so the human scores on the judge's scale."""
    records = await _records(db, id_)
    idx_by_v: dict = {}
    for i, r in enumerate(records):
        idx_by_v.setdefault(r.get("verdict"), []).append(i)
    rng = random.Random(0)
    per = max(1, n // max(1, len(idx_by_v)))
    picked = []
    for v, idxs in idx_by_v.items():
        rng.shuffle(idxs)
        picked += idxs[:per]
    picked.sort()
    study = await db.get(models.Study, id_)
    items = [{"key": i, "question": records[i].get("question") or records[i].get("input_id"),
              "reference": records[i].get("reference"), "answer": records[i].get("answer")}
             for i in picked]
    return {"rubric": await _study_rubric(db, id_),
            "judge_model": study.judge_model if study else "",
            "items": items}


class HumanRatingsIn(BaseModel):
    rater: str
    scores: dict[str, float]   # {answer_key: score}


@router.post("/studies/{id_}/human-ratings", status_code=201)
async def submit_ratings(id_: int, body: HumanRatingsIn, db: AsyncSession = Depends(get_session)):
    # replace this rater's prior scores for the study
    existing = (await db.execute(select(models.HumanRating).where(
        models.HumanRating.study_id == id_, models.HumanRating.rater == body.rater))).scalars()
    for row in existing:
        await db.delete(row)
    for key, score in body.scores.items():
        db.add(models.HumanRating(study_id=id_, answer_key=str(key), rater=body.rater, score=float(score)))
    await db.commit()
    return {"saved": len(body.scores)}


@router.get("/studies/{id_}/raters")
async def list_raters(id_: int, db: AsyncSession = Depends(get_session)):
    rows = (await db.execute(select(models.HumanRating).where(models.HumanRating.study_id == id_))).scalars()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.rater] = counts.get(r.rater, 0) + 1
    return [{"rater": k, "n": v} for k, v in counts.items()]


async def _study_metric(db: AsyncSession, study_id: int) -> str:
    study = await db.get(models.Study, study_id)
    if study and study.rubric_id:
        rubric = await db.get(models.Rubric, study.rubric_id)
        if rubric:
            return _METRIC.get(rubric.scale_type, "ordinal")
    return "ordinal"


def _alpha(raters: dict, metric: str):
    r = cafe.reliability(raters=raters, metric=metric)
    return {"alpha": None if r.alpha != r.alpha else round(r.alpha, 3),
            "n_units": r.n_units, "raters": list(r.raters),
            "interpret": r.interpret(r.alpha) if r.n_units >= 2 else "undefined"}


@router.get("/studies/{id_}/reliability")
async def reliability(id_: int, db: AsyncSession = Depends(get_session)):
    """Human↔human (the ceiling) and judge↔human Krippendorff's α."""
    records = await _records(db, id_)
    metric = await _study_metric(db, id_)

    rows = list((await db.execute(select(models.HumanRating).where(
        models.HumanRating.study_id == id_))).scalars())
    humans: dict[str, dict] = {}
    for r in rows:
        humans.setdefault(r.rater, {})[r.answer_key] = r.score
    if not humans:
        raise HTTPException(400, "no human ratings yet")

    judge = {str(i): rec.get("verdict") for i, rec in enumerate(records) if rec.get("verdict") is not None}

    out = {"metric": metric, "n_human_raters": len(humans)}
    if len(humans) >= 2:
        out["human_ceiling"] = _alpha(humans, metric)
    out["judge_vs_human"] = _alpha({"judge": judge, **humans}, metric)
    return out
