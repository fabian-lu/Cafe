"""One-off importer: insert a study rebuilt from checkpoints (see the host-side
recover_build.py) into the DB, so an expensive run recovered offline shows up in the UI
exactly like a fresh run — same Study + Rubric + Dataset + cached StudyResult.

    python -m app.recover /tmp/recovered_payload.json                 # insert a new study
    python -m app.recover /tmp/recovered_payload.json --replace 3     # refresh study 3's cached payload
"""
from __future__ import annotations

import asyncio
import json
import sys

from sqlalchemy import select

from app.db import SessionLocal, init_db
from app import models


async def _replace_payload(path: str, study_id: int) -> None:
    """Update just an existing study's cached StudyResult.payload (e.g. after adding a field like
    timing to the payload) — leaves the Study/Rubric/Dataset rows untouched."""
    blob = json.load(open(path, encoding="utf-8"))
    async with SessionLocal() as db:
        res = (await db.execute(select(models.StudyResult).where(
            models.StudyResult.study_id == study_id))).scalar_one_or_none()
        if res is None:
            raise SystemExit(f"study {study_id} has no StudyResult to replace")
        res.payload = blob["payload"]
        await db.commit()
        print(f"replaced payload for study {study_id} "
              f"(records={len(blob['payload'].get('records', []))}, timing={'yes' if blob['payload'].get('timing') else 'no'})")


async def _run(path: str) -> None:
    blob = json.load(open(path, encoding="utf-8"))
    await init_db()
    async with SessionLocal() as db:
        rubric = models.Rubric(**blob["rubric"])
        dataset = models.Dataset(**blob["dataset"])
        db.add_all([rubric, dataset])
        await db.flush()  # assign ids

        s = blob["study"]
        study = models.Study(
            name=s["name"], pipeline=s.get("pipeline", "pipeline"), factors=s["factors"],
            dataset_id=dataset.id, rubric_id=rubric.id, judge_model=s.get("judge_model", ""),
            replications=s.get("replications", 1), status="done", progress=100,
        )
        db.add(study)
        await db.flush()

        db.add(models.StudyResult(study_id=study.id, payload=blob["payload"]))
        await db.commit()
        print(f"recovered study id={study.id} '{study.name}' "
              f"({len(blob['payload'].get('records', []))} records) → rubric {rubric.id}, dataset {dataset.id}")


if __name__ == "__main__":
    _path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/recovered_payload.json"
    if "--replace" in sys.argv:
        _sid = int(sys.argv[sys.argv.index("--replace") + 1])
        asyncio.run(_replace_payload(_path, _sid))
    else:
        asyncio.run(_run(_path))
