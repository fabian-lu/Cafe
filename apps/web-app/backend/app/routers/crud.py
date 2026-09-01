"""CRUD endpoints for rubrics, datasets (questions), and studies."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
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


def _rubric_row(ev) -> models.Rubric:
    """A Rubric row from an Evaluation's ratings (judge system prompt lives on the ratings)."""
    rb = ev.ratings.rubric
    return models.Rubric(
        name=rb.name,
        scale_type=rb.scale_type.value,
        levels=[{"value": lv.value, "label": lv.label, "description": lv.description} for lv in rb.levels],
        instruction=rb.instruction,
        prompt_template=rb.prompt_template,
        system_prompt=ev.ratings.judge_system_prompt,
        preset="reference_qa",
    )


async def _attach_dimension(db: AsyncSession, study: models.Study, ev, payload) -> None:
    """Add one Evaluation as an additional dimension (Rubric + StudyResult row) of ``study``."""
    rb_name = ev.ratings.rubric.name
    dupe = (await db.execute(select(models.StudyResult).where(
        models.StudyResult.study_id == study.id,
        models.StudyResult.dimension == rb_name))).scalars().first()
    if dupe is not None:
        raise HTTPException(409, f"study {study.id} already has a dimension {rb_name!r} — "
                                 "rename the rubric or delete that dimension first")
    rubric = _rubric_row(ev)
    db.add(rubric)
    await db.flush()
    study.rubric_ids = [*(study.rubric_ids or ([study.rubric_id] if study.rubric_id else [])),
                        rubric.id]
    db.add(models.StudyResult(study_id=study.id, dimension=rb_name, payload=payload))


@router.post("/studies/import", response_model=schemas.StudyOut, status_code=201)
async def import_study(request: Request, attach_to: int | None = None,
                       db: AsyncSession = Depends(get_session)):
    """Import a WHOLE study from a notebook / offline run — no re-running, no pipeline required.

    The JSON body is either a single ``cafe.save_evaluation`` bundle, or a **multi-dimension
    bundle** ``{"cafe_multi_dimension_bundle": 1, "name": ..., "dimensions": {<name>: <bundle>}}``
    (the same answers judged on several rubrics, e.g. via ``result.rejudge``). One POST creates
    the study with ALL its dimensions; the Results page then offers a dimension selector.

    ``?attach_to=<study_id>`` adds the bundle's dimension(s) to an existing study instead of
    creating a new one (rarely needed — e.g. re-judging an old study on a new rubric later).
    """
    import cafe

    from app.runner import _results_payload

    raw_body = await request.body()
    try:
        raw = json.loads(raw_body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"not valid JSON: {exc}")

    name_override = None
    if isinstance(raw, dict) and isinstance(raw.get("dimensions"), dict) and raw["dimensions"]:
        name_override = raw.get("name")
        bundles = list(raw["dimensions"].values())
    else:
        bundles = [raw]

    evs = []
    for b in bundles:
        try:
            ev = cafe.load_evaluation(b)
        except Exception as exc:  # noqa: BLE001 — surface a clean 400 for a bad upload
            raise HTTPException(400, f"not a valid cafe evaluation bundle: {exc}")
        if ev.ratings is None:
            raise HTTPException(400, "an evaluation in this bundle has no judge ratings — "
                                     "nothing to attribute")
        evs.append(ev)
    payloads = [_results_payload(ev) for ev in evs]

    if attach_to is not None:
        study = await db.get(models.Study, attach_to)
        if study is None:
            raise HTTPException(404, f"study {attach_to} not found")
        for ev, payload in zip(evs, payloads):
            await _attach_dimension(db, study, ev, payload)
        await db.commit()
        await db.refresh(study)
        return study

    # Create the study from the FIRST dimension; the rest attach to it.
    first, first_payload = evs[0], payloads[0]
    rubric = _rubric_row(first)
    ids = list(first.questions) or sorted({o.input_id for o in first.answers.observations})
    dataset = models.Dataset(
        name=f"{name_override or first.study_name or 'imported'} (imported)",
        items=[{"id": i, "text": first.questions.get(i, ""), "reference": first.references.get(i)}
               for i in ids],
    )
    db.add_all([rubric, dataset])
    await db.flush()  # assign ids

    # Factors + their observed levels, read off the answer configs (order preserved).
    names = list(first.ratings.factors) or list(
        first.answers.observations[0].config if first.answers.observations else {})
    factors = []
    for name in names:
        levels: list = []
        for o in first.answers.observations:
            v = o.config.get(name)
            if v not in levels:
                levels.append(v)
        factors.append({"name": name, "levels": levels})
    reps = max((o.rep for o in first.answers.observations), default=0) + 1

    study = models.Study(
        name=name_override or first.study_name or "imported study",
        description="Imported from a notebook / offline run.",
        pipeline="(imported)", factors=factors, dataset_id=dataset.id, rubric_id=rubric.id,
        rubric_ids=[rubric.id],
        judge_model=first.ratings.judge_model, replications=reps, status="done", progress=100,
    )
    db.add(study)
    await db.flush()
    db.add(models.StudyResult(study_id=study.id, dimension=first.ratings.rubric.name,
                              payload=first_payload))
    for ev, payload in zip(evs[1:], payloads[1:]):
        await _attach_dimension(db, study, ev, payload)
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
