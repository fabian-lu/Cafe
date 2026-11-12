"""Fixtures for the backend tests: each test gets a backend with a database of its own.

The backend reads its database from ``DATABASE_URL`` when ``app.config`` is imported, so the
``backend`` fixture points that variable at a fresh SQLite file in the test's temporary directory
and imports the ``app`` package anew. :meth:`Backend.restart` does what restarting the backend
process does: it shuts the app down, drops every module-level state (the in-memory caches, run
progress) by importing the app again, and starts it on the same database file.

Nothing here calls an LLM or the network. The filtered-view recompute is counted, and unless a
test asks for the real statistics it is replaced by a stand-in that answers at once.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import random
import sys
import time
from pathlib import Path

import cafe
import httpx
import pytest_asyncio
from sqlalchemy import select

# A pipeline without LLM calls, so a test can run (and re-run) a study for real.
TOY_PIPELINE = '''
import cafe

pipe = cafe.Pipeline()


@pipe.technique("reply", "short")
async def short(ctx, query):
    return query.split()[0]


@pipe.technique("reply", "long")
async def long(ctx, query):
    return query


@pipe.compose
async def run(config, item, ctx):
    return await ctx.run("reply", query=item["text"])
'''

# Questions in two topics: "topic" is what the tests filter by (GET /filters offers a and b).
ITEMS = [{"id": f"q{i}", "text": f"Question number {i}?", "reference": f"Answer {i}.",
          "topic": "ab"[i % 2]} for i in range(8)]

NUMERIC_1_5 = {"name": "quality", "scale_type": "numeric",
               "levels": [{"value": v, "label": str(v)} for v in range(1, 6)]}


def _forget_app_modules() -> None:
    for name in [n for n in sys.modules if n == "app" or n.startswith("app.")]:
        del sys.modules[name]


class Backend:
    """One backend process (the FastAPI app, in-process) and helpers to drive it over HTTP.

    Attributes
    ----------
    computed : list of tuple
        ``(fkey, fval)`` of every filtered view the backend computed, across restarts.
    real_statistics : bool
        Compute filtered views with cafe-core instead of the stand-in.
    compute_delay : float
        Seconds the stand-in takes (in the worker thread, like the real refit).
    stats_version_bump : int
        Added to ``runner.STATS_VERSION`` on start: a backend with newer statistics.
    """

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.computed: list[tuple[str, str]] = []
        self.real_statistics = False
        self.compute_delay = 0.0
        self.stats_version_bump = 0
        self.running = False

    async def start(self) -> None:
        _forget_app_modules()
        self.main = importlib.import_module("app.main")
        self.db = importlib.import_module("app.db")
        self.models = importlib.import_module("app.models")
        self.runner = importlib.import_module("app.runner")
        importlib.import_module("app.seed").SEED_FILE = self.tmp_path / "no-demo-seed.json"
        self.runner.STATS_VERSION += self.stats_version_bump
        real = self.runner.filtered_payload

        def counting(payload, items, fkey, fval):
            self.computed.append((fkey, fval))
            if self.real_statistics:
                return real(payload, items, fkey, fval)
            time.sleep(self.compute_delay)
            return {"filtered": {"key": fkey, "value": fval},
                    "computed_from": payload.get("computed_at"),
                    "computation": len(self.computed)}

        self.runner.filtered_payload = counting
        self._lifespan = self.main.lifespan(self.main.app)
        await self._lifespan.__aenter__()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.main.app),
                                        base_url="http://backend")
        self.running = True

    async def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        await self.client.aclose()
        await self._lifespan.__aexit__(None, None, None)
        await self.db.engine.dispose()

    async def restart(self) -> None:
        await self.stop()
        await self.start()

    @property
    def views(self):
        """The ``app.filtered_views`` module of the running backend."""
        return importlib.import_module("app.filtered_views")

    # ── HTTP ──────────────────────────────────────────────────────────────────
    async def get(self, path: str, **params) -> dict:
        response = await self.client.get(path, params=params)
        assert response.status_code == 200, response.text
        return response.json()

    async def view(self, study_id: int, fval: str = "a", fkey: str = "topic", **params) -> dict:
        """A filtered results view (``GET /results?fkey&fval``)."""
        return await self.get(f"/api/studies/{study_id}/results", fkey=fkey, fval=fval, **params)

    async def run(self, study_id: int) -> None:
        """Run the study (``POST /run``) and wait until it has finished."""
        response = await self.client.post(f"/api/studies/{study_id}/run")
        assert response.status_code == 200, response.text
        for _ in range(600):
            status = (await self.get(f"/api/studies/{study_id}"))["status"]
            if status != "running":
                assert status == "done", self.runner.PROGRESS.get(study_id)
                return
            await asyncio.sleep(0.05)
        raise AssertionError("the run did not finish within 30 s")

    # ── database ──────────────────────────────────────────────────────────────
    async def add_study(self, *, judged: bool = True, dimensions: tuple[str, ...] = ("quality",),
                        computed_at: float = 1000.0) -> int:
        """Insert a finished study whose stored results have real records (two models, the
        questions of :data:`ITEMS`), without fitting any statistics. Returns its id."""
        records = [{"input_id": it["id"], "question": it["text"], "reference": it["reference"],
                    "output": f"{model} answers {it['id']}", "model": model, "rep": 0,
                    "judge_rep": 0, "elapsed_s": 0.1, "cost_usd": 0.001,
                    "verdict": (3 + (model == "large") + i % 2) if judged else None,
                    "reasoning": "fine" if judged else None}
                   for model in ("small", "large") for i, it in enumerate(ITEMS)]
        async with self.db.SessionLocal() as session:
            dataset = self.models.Dataset(name="questions", items=ITEMS)
            session.add(dataset)
            await session.flush()
            study = self.models.Study(name="study", pipeline="toy", dataset_id=dataset.id,
                                      factors=[{"name": "model", "levels": ["small", "large"]}],
                                      status="done", progress=100)
            session.add(study)
            await session.flush()
            for dimension in dimensions:
                session.add(self.models.StudyResult(study_id=study.id, dimension=dimension, payload={
                    "factors": ["model"], "records": records,
                    "rubric": {**NUMERIC_1_5, "name": dimension} if judged else None,
                    "stats_version": self.runner.STATS_VERSION, "computed_at": computed_at}))
            await session.commit()
            return study.id

    async def result_ids(self, study_id: int) -> list[int]:
        StudyResult = self.models.StudyResult
        async with self.db.SessionLocal() as session:
            return list((await session.execute(select(StudyResult.id).where(
                StudyResult.study_id == study_id).order_by(StudyResult.id))).scalars())

    async def set_payload(self, result_id: int, payload: dict) -> None:
        """Replace a StudyResult's payload directly in the database (no app code involved)."""
        async with self.db.SessionLocal() as session:
            (await session.get(self.models.StudyResult, result_id)).payload = payload
            await session.commit()

    async def stored_views(self) -> list[tuple[int, str, str, str]]:
        """``(study_result_id, fkey, fval, version)`` of every row in ``filtered_results``."""
        FilteredResult = self.models.FilteredResult
        async with self.db.SessionLocal() as session:
            rows = await session.execute(select(
                FilteredResult.study_result_id, FilteredResult.fkey, FilteredResult.fval,
                FilteredResult.version))
            return sorted(tuple(row) for row in rows)


@pytest_asyncio.fixture
async def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'cafe.db'}")
    techniques = tmp_path / "techniques"
    techniques.mkdir()
    (techniques / "toy.py").write_text(TOY_PIPELINE)
    monkeypatch.setenv("CAFE_TECHNIQUES_DIR", str(techniques))
    server = Backend(tmp_path)
    await server.start()
    yield server
    await server.stop()
    _forget_app_modules()


def judged_bundle(tmp_path: Path) -> dict:
    """A small judged study as a ``cafe.save_evaluation`` bundle: 2 x 2 configurations over the
    questions of :data:`ITEMS`, scored 1-5 by a made-up judge (a numeric scale: no R needed)."""
    rng = random.Random(0)
    factors = ["model", "prompt"]
    observations, ratings = [], []
    for model in ("small", "large"):
        for prompt in ("plain", "cot"):
            config = {"model": model, "prompt": prompt}
            for it in ITEMS:
                obs = cafe.Observation(config=config, input_id=it["id"], rep=0,
                                       output=f"{model}/{prompt}: {it['id']}", elapsed_s=0.1,
                                       metadata={"cost_usd": 0.002 if model == "large" else 0.001})
                score = round(2 + (model == "large") + 0.5 * (prompt == "cot") + rng.gauss(0, 0.7))
                score = min(5, max(1, score))
                observations.append(obs)
                ratings.append(cafe.Rating(obs_key=obs.key(), config=config, input_id=it["id"],
                                           rep=0, judge_rep=0, value=score, value_numeric=score,
                                           reasoning="made up"))
    rubric = cafe.Rubric(name="quality", scale_type=cafe.ScaleType.numeric,
                         levels=[cafe.Level(v, str(v), "") for v in range(1, 6)],
                         instruction="Rate the answer.")
    evaluation = cafe.Evaluation(
        study_name="synthetic",
        answers=cafe.Results(study_name="synthetic", factors=factors, observations=observations),
        ratings=cafe.Ratings(rubric=rubric, judge_model="made-up-judge", factors=factors,
                             items=ratings),
        questions={it["id"]: it["text"] for it in ITEMS},
        references={it["id"]: it["reference"] for it in ITEMS},
        items=ITEMS)
    path = tmp_path / "bundle.json"
    cafe.save_evaluation(evaluation, str(path))
    return json.loads(path.read_text())
