"""CRUD endpoints for rubrics, datasets (questions), and studies."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete as sql_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import models, schemas
from app.db import get_session

router = APIRouter(prefix="/api", tags=["crud"])


async def _get_or_404(db: AsyncSession, model, id_: int):
    obj = await db.get(model, id_)
    if obj is None:
        raise HTTPException(404, f"{model.__name__} {id_} not found")
    return obj


# ── Rubrics ──────────────────────────────────────────────────────────────────
@router.get("/rubrics", response_model=list[schemas.RubricOut])
async def list_rubrics(db: AsyncSession = Depends(get_session)):
    return list((await db.execute(select(models.Rubric).order_by(models.Rubric.id))).scalars())


@router.get("/rubrics/{id_}", response_model=schemas.RubricOut)
async def get_rubric(id_: int, db: AsyncSession = Depends(get_session)):
    return await _get_or_404(db, models.Rubric, id_)


@router.post("/rubrics", response_model=schemas.RubricOut, status_code=201)
async def create_rubric(body: schemas.RubricIn, db: AsyncSession = Depends(get_session)):
    obj = models.Rubric(**body.model_dump())
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def _in_use(db: AsyncSession, col, id_: int) -> bool:
    return (await db.execute(select(models.Study.id).where(col == id_))).first() is not None


@router.delete("/rubrics/{id_}", status_code=204)
async def delete_rubric(id_: int, db: AsyncSession = Depends(get_session)):
    await _get_or_404(db, models.Rubric, id_)
    if await _in_use(db, models.Study.rubric_id, id_):
        raise HTTPException(409, "rubric is used by a study — delete the study first")
    await db.execute(sql_delete(models.Rubric).where(models.Rubric.id == id_))
    await db.commit()


# ── Datasets (questions) ─────────────────────────────────────────────────────
@router.get("/datasets", response_model=list[schemas.DatasetOut])
async def list_datasets(db: AsyncSession = Depends(get_session)):
    return list((await db.execute(select(models.Dataset).order_by(models.Dataset.id))).scalars())


@router.get("/datasets/{id_}", response_model=schemas.DatasetOut)
async def get_dataset(id_: int, db: AsyncSession = Depends(get_session)):
    return await _get_or_404(db, models.Dataset, id_)


@router.post("/datasets", response_model=schemas.DatasetOut, status_code=201)
async def create_dataset(body: schemas.DatasetIn, db: AsyncSession = Depends(get_session)):
    obj = models.Dataset(**body.model_dump())
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


@router.delete("/datasets/{id_}", status_code=204)
async def delete_dataset(id_: int, db: AsyncSession = Depends(get_session)):
    await _get_or_404(db, models.Dataset, id_)
    if await _in_use(db, models.Study.dataset_id, id_):
        raise HTTPException(409, "dataset is used by a study — delete the study first")
    await db.execute(sql_delete(models.Dataset).where(models.Dataset.id == id_))
    await db.commit()


# ── Studies ──────────────────────────────────────────────────────────────────
@router.get("/studies", response_model=list[schemas.StudyOut])
async def list_studies(archived: bool = False, db: AsyncSession = Depends(get_session)):
    """Live studies by default; pass ``?archived=true`` for the archived (soft-deleted) ones."""
    q = select(models.Study).where(models.Study.archived == archived).order_by(models.Study.id.desc())
    return list((await db.execute(q)).scalars())


@router.get("/studies/{id_}", response_model=schemas.StudyOut)
async def get_study(id_: int, db: AsyncSession = Depends(get_session)):
    return await _get_or_404(db, models.Study, id_)


@router.post("/studies", response_model=schemas.StudyOut, status_code=201)
async def create_study(body: schemas.StudyIn, db: AsyncSession = Depends(get_session)):
    obj = models.Study(**body.model_dump(), status="draft", progress=0)
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


@router.post("/studies/{id_}/archive", response_model=schemas.StudyOut)
async def archive_study(id_: int, db: AsyncSession = Depends(get_session)):
    """Soft-delete: hide the study but keep its results/ratings (the default 'delete' action)."""
    study = await _get_or_404(db, models.Study, id_)
    study.archived = True
    await db.commit()
    await db.refresh(study)
    return study


@router.post("/studies/{id_}/restore", response_model=schemas.StudyOut)
async def restore_study(id_: int, db: AsyncSession = Depends(get_session)):
    study = await _get_or_404(db, models.Study, id_)
    study.archived = False
    await db.commit()
    await db.refresh(study)
    return study


@router.delete("/studies/{id_}", status_code=204)
async def delete_study(id_: int, db: AsyncSession = Depends(get_session)):
    """Permanent delete — destroys the run's results/ratings. Guarded: a study must be archived
    first, so an expensive run can never be lost with a single accidental click."""
    study = await _get_or_404(db, models.Study, id_)
    if not study.archived:
        raise HTTPException(409, "archive the study before permanently deleting it")
    # remove dependent rows first (avoids FK violations), then the study
    await db.execute(sql_delete(models.StudyResult).where(models.StudyResult.study_id == id_))
    await db.execute(sql_delete(models.HumanRating).where(models.HumanRating.study_id == id_))
    await db.execute(sql_delete(models.Answer).where(models.Answer.study_id == id_))
    await db.execute(sql_delete(models.Study).where(models.Study.id == id_))
    await db.commit()
