# CAFE Platform (web app)

The interactive web platform that wraps `cafe-core`: define factorial studies over your compound-AI
system, launch them, watch progress, explore results, collect human ratings. See the
[web app guide](../../docs/guides/web-app.md) for a walkthrough.

## Run it (Docker)

```bash
cd apps/web-app
cp ../../.env.example ../../.env   # add your OLLAMA_API_KEY / OPENROUTER_API_KEY to the repo-root .env
docker compose up
```

- Frontend → http://localhost:5173
- Backend API → http://localhost:8000/api/health, http://localhost:8000/api/pipelines

## The system under test

Lives in code, in the repo-root [`techniques/`](../../techniques) folder (mounted into the backend).
Every `.py` file there whose module-level `pipe` (a `cafe.Pipeline`) has a `@pipe.compose` is a
selectable pipeline, named after the file; files without a compose are technique libraries. The app
discovers them at startup and builds the study form's factors from the chosen pipeline's
stages/techniques. Edit that folder (or point `CAFE_TECHNIQUES_DIR` / the docker volume at your own)
and hit **Reload** on the Techniques page to pick up changes.

## Local dev (without Docker)

```bash
# backend
cd apps/web-app/backend && pip install -r requirements.txt && pip install -e ../../../packages/cafe-core
uvicorn app.main:app --reload
# frontend (another shell)
cd apps/web-app/frontend && npm install && npm run dev
```

## Status

All pages are implemented:

- **Studies** — create a study over one of the discovered pipelines, optionally with extra rubrics
  (each judges the same answers as its own dimension); estimate time/cost, run it with live progress,
  archive/restore/delete. **Import study** loads a `cafe.save_evaluation` bundle (or a
  multi-dimension bundle) without re-running anything.
- **Questions / Rubrics** — create, view and delete question datasets and rubrics.
- **Techniques** — the discovered pipelines with their stages, techniques and parameters.
- **Results** — attribution, marginal means, interactions, the configuration ranking, effect sizes,
  the ordinal or binary model (by rubric scale), variance components, latency, verdict distribution, run timing and
  cost/quality. Multi-rubric studies get a dimension selector. Extra question fields (2–30 distinct
  values) become filters that recompute the statistics on that subset of questions. Energy is shown
  when the pipeline's techniques declare coefficients.
- **Raters** — score a study's answers by name on the judge's rubric and see judge↔human and
  human↔human Krippendorff's α (on the primary dimension).
