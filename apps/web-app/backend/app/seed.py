"""Seed an empty database with the bundled demo study (the HotpotQA RAG evaluation), so a fresh
``docker compose up`` opens straight into a real study with full results — no run, no API keys needed.

It is a **no-op** if the database already has any study (so it never touches an existing install) or
if the seed file is missing. The seed is the same ``{study, rubric, dataset, payload}`` shape the
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
    """Insert the demo study iff the DB has no studies yet and the seed file exists."""
    if not SEED_FILE.exists():
        return
    async with SessionLocal() as db:
        count = (await db.execute(select(func.count(models.Study.id)))).scalar_one()
        if count:
            return  # existing install — leave it alone

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
