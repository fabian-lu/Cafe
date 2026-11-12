"""One-off importer: insert a study rebuilt from checkpoints (see the host-side
recover_build.py) into the DB, so an expensive run recovered offline shows up in the UI
exactly like a fresh run — same Study + Rubric + Dataset + cached StudyResult.

    python -m app.recover /tmp/recovered_payload.json                 # insert a new study
    python -m app.recover /tmp/recovered_payload.json --replace 3     # refresh study 3's cached payload
    python -m app.recover /tmp/recovered_payload.json --replace 3 --dimension tone  # ...of one dimension
"""
from __future__ import annotations

import asyncio
import json
import sys

from sqlalchemy import select

from app.db import SessionLocal, init_db
from app import models


async def _replace_payload(path: str, study_id: int, dimension: str | None = None) -> None:
    """Update just an existing study's cached StudyResult.payload (e.g. after adding a field like
    timing to the payload) — leaves the Study/Rubric/Dataset rows untouched.

    Parameters
    ----------
    path : str
        A recovered-payload JSON file (``{"payload": ...}``).
    study_id : int
        The study whose cached results are replaced.
    dimension : str, optional
        The dimension (rubric name) whose results are replaced; a study judged on several
        rubrics has one StudyResult per dimension. Default: the primary one, its first row,
        which the Results page shows first.

    Raises
    ------
    SystemExit
        If the study has no StudyResult, or none for ``dimension`` (the message names the
        dimensions it has).
    """
    blob = json.load(open(path, encoding="utf-8"))
    async with SessionLocal() as db:
        rows = list((await db.execute(select(models.StudyResult).where(
            models.StudyResult.study_id == study_id)
            .order_by(models.StudyResult.id))).scalars())
        res = next((r for r in rows if dimension is None or r.dimension == dimension), None)
        if res is None:
            if not rows:
                raise SystemExit(f"study {study_id} has no StudyResult to replace")
            raise SystemExit(f"study {study_id} has no dimension {dimension!r}; its dimensions: "
                             + ", ".join(repr(r.dimension) for r in rows))
        res.payload = blob["payload"]
        await db.commit()
        print(f"replaced payload for study {study_id}, dimension {res.dimension!r} "
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
    _dim = sys.argv[sys.argv.index("--dimension") + 1] if "--dimension" in sys.argv else None
    if "--replace" in sys.argv:
        _sid = int(sys.argv[sys.argv.index("--replace") + 1])
        asyncio.run(_replace_payload(_path, _sid, _dim))
    elif _dim is not None:
        raise SystemExit("--dimension picks the results that --replace replaces; it needs --replace")
    else:
        asyncio.run(_run(_path))
