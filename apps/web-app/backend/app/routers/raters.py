"""Human rating + inter-rater reliability (Krippendorff's α).

A study's answers (from the cached results) are sampled into a rating sheet; humans score them by
name (no accounts). Each score is stored under the answer's key — cafe's observation key plus a
digest of the answer text — so it stays with the exact text that was rated (a re-run regenerates
the answers under the same observation keys). cafe-core aligns the scores with the judge's
verdicts per answer and computes α.
"""

from __future__ import annotations

import hashlib
import json
import random

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import cafe
from cafe.execution.results import Observation

from app import models, runner
from app.db import get_session

router = APIRouter(prefix="/api", tags=["raters"])

_METRIC = {"ordinal": "ordinal", "numeric": "interval", "binary": "nominal"}


async def _primary_payload(db: AsyncSession, study_id: int) -> dict | None:
    # Multi-dimension studies have one StudyResult per rubric; human rating + IRR operate on
    # the PRIMARY dimension (the first row) — per-dimension human rating is future work.
    res = (await db.execute(select(models.StudyResult).where(
        models.StudyResult.study_id == study_id)
        .order_by(models.StudyResult.id))).scalars().first()
    return None if res is None else res.payload


async def _results(db: AsyncSession, study_id: int) -> dict:
    payload = await _primary_payload(db, study_id)
    if payload is None:
        raise HTTPException(404, "no results yet — run the study first")
    return payload


def _answer_key(rec: dict, factors: list[str]) -> str:
    """The key a human score is stored under: cafe's observation key (config · input · rep)
    plus a digest of the answer text. Re-running the study regenerates the answers under the
    same observation keys, so the digest is what ties a score to the text that was rated."""
    obs = Observation(config={f: rec.get(f) for f in factors}, input_id=rec.get("input_id"),
                      rep=int(rec.get("rep") or 0))
    text = rec["output"] if "output" in rec else rec.get("answer")  # older payloads: "answer"
    digest = hashlib.sha1(json.dumps(text, default=str).encode("utf-8")).hexdigest()[:12]
    return f"{obs.key()}::{digest}"


def _answer_keys(payload: dict) -> list[str]:
    factors = list(payload.get("factors") or [])
    return [_answer_key(r, factors) for r in payload.get("records", [])]


def _current_key(stored: str, keys: list[str]) -> str | None:
    """Map a stored ``answer_key`` onto the current answers' keys (``None`` = stale). Ratings
    saved before answer keys existed hold the row index into the records list; they keep that
    meaning until ``pin_legacy_rating_keys`` converts them."""
    if stored.isdigit():
        return keys[int(stored)] if int(stored) < len(keys) else None
    return stored if stored in keys else None


async def pin_legacy_rating_keys(db: AsyncSession, study_id: int) -> None:
    """Convert this study's index-keyed (legacy) human ratings to answer keys of the CURRENT
    results. The run task calls this right before a re-run replaces those results, so a legacy
    score stays with the answer that was rated instead of the new answer at its index."""
    rows = [r for r in (await db.execute(select(models.HumanRating).where(
        models.HumanRating.study_id == study_id))).scalars() if r.answer_key.isdigit()]
    payload = await _primary_payload(db, study_id) if rows else None
    if payload is None:
        return
    keys = _answer_keys(payload)
    for row in rows:
        row.answer_key = _current_key(row.answer_key, keys) or f"unmatched:{row.answer_key}"


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
    payload = await _results(db, id_)
    records, keys = payload.get("records", []), _answer_keys(payload)
    first: dict[str, int] = {}  # one item per answer (judge replications share an answer key)
    for i, k in enumerate(keys):
        first.setdefault(k, i)
    idx_by_v: dict = {}
    for i in first.values():
        idx_by_v.setdefault(records[i].get("verdict"), []).append(i)
    rng = random.Random(0)
    per = max(1, n // max(1, len(idx_by_v)))
    picked = []
    for v, idxs in idx_by_v.items():
        rng.shuffle(idxs)
        picked += idxs[:per]
    picked.sort()
    study = await db.get(models.Study, id_)
    items = [{"key": keys[i], "question": records[i].get("question") or records[i].get("input_id"),
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
    """Human↔human (the ceiling) and judge↔human Krippendorff's α, over the current answers.

    Scores whose answer text is no longer in the results (the study was re-run since) are left
    out and counted in ``stale_ratings``."""
    payload = await _results(db, id_)
    metric = await _study_metric(db, id_)

    rows = list((await db.execute(select(models.HumanRating).where(
        models.HumanRating.study_id == id_))).scalars())
    if not rows:
        raise HTTPException(400, "no human ratings yet")
    keys = _answer_keys(payload)
    humans: dict[str, dict] = {}
    stale = 0
    for r in rows:
        scores = humans.setdefault(r.rater, {})
        key = _current_key(r.answer_key, keys)
        if key is None:
            stale += 1
        else:
            scores[key.rsplit("::", 1)[0]] = r.score  # the cafe obs key the judge is keyed on
    # the judge as a cafe Evaluation: cafe-core collapses its verdicts to one score per answer
    records = payload.get("records", [])
    judge = runner._evaluation_from_payload(payload, [], {r.get("input_id") for r in records})

    out = {"metric": metric, "n_human_raters": len(humans), "stale_ratings": stale}
    if len(humans) >= 2:
        out["human_ceiling"] = _alpha(humans, metric)
    out["judge_vs_human"] = _alpha({"judge": judge, **humans}, metric)
    return out
