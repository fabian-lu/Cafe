"""Seed an empty database with the bundled demo study (the HotpotQA RAG evaluation), so a fresh
``docker compose up`` opens straight into a real study with full results — no run, no API keys needed.

It only acts on a **fresh** database, one without any study, rubric or dataset: it never touches an
existing install, and a demo study the user deleted doesn't come back on the next start (docker
compose runs the backend with ``--reload``, so every code edit is a restart). It is also a no-op if
the seed file is missing. The seed is the same ``{study, rubric, dataset, payload}`` shape the
``app.recover`` importer uses.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import func, select

from app import models
from app.db import SessionLocal

SEED_FILE = Path(__file__).parent / "seed" / "demo_seed.json"


async def seed_if_empty() -> None:
    """Insert the demo study (with its rubric and dataset) into a fresh database.

    A database is fresh when it holds no study, rubric or dataset at all; nothing happens
    otherwise, or if the seed file is missing. No studies alone isn't enough: once the user
    deletes the demo study its rubric and dataset remain, and every later start would add the
    demo back, with another copy of each.
    """
    if not SEED_FILE.exists():
        return
    async with SessionLocal() as db:
        for model in (models.Study, models.Rubric, models.Dataset):
            if (await db.execute(select(func.count(model.id)))).scalar_one():
                return  # not a fresh database — leave it alone

        blob = json.loads(SEED_FILE.read_text(encoding="utf-8"))
        rubric = models.Rubric(**blob["rubric"])
        dataset = models.Dataset(**blob["dataset"])
        db.add_all([rubric, dataset])
        await db.flush()  # assign ids

        s = blob["study"]
        study = models.Study(
            name=s["name"],
            description=s.get("description", ""),
            pipeline=s.get("pipeline", "pipeline"),
            factors=s["factors"],
            dataset_id=dataset.id,
            rubric_id=rubric.id,
            judge_model=s.get("judge_model", ""),
            replications=s.get("replications", 1),
            concurrency=s.get("concurrency", 8),
            status="done",
            progress=100,
        )
        db.add(study)
        await db.flush()
        db.add(models.StudyResult(study_id=study.id, payload=blob["payload"]))
        await db.commit()
        print(f"[seed] inserted demo study '{study.name}' "
              f"({len(blob['payload'].get('records', []))} records)")
