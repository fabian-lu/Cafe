# The web app

CAFE ships a self-hostable web platform over the **same engine** as the library: define
a study in the browser, run it with live progress, and explore the full analytics —
then collect human ratings. Anything you can do here you can also do in Python; the app
just wraps `cafe-core` behind a FastAPI (JSON) API and a React front-end, backed by
Postgres. No data leaves your machine.

Prefer to look before installing? The [**live demo**](https://cafe-ai.de/demo) is a
read-only snapshot of a finished study.

## Run it

The platform runs entirely in containers — you only need **Docker + Docker Compose**
on the host (Python, R, Node, and Postgres all run inside):

```bash
cd apps/web-app
cp ../../.env.example ../../.env     # add your LLM keys to the repo-root .env
docker compose up
```

Then open **http://localhost:5173**. The first build takes a few minutes (it installs R
and the front-end deps); subsequent starts are fast. Keys come from the repo-root
`.env` — the same file the library and CLI use.

## Your system under test

The app evaluates the pipeline in the repo-root [`techniques/`](https://github.com/fabian-lu/Cafe/tree/main/techniques)
folder, which is mounted into the backend. It must expose a module-level `pipe` (a
`cafe.Pipeline`) in `pipeline.py`; the app discovers it at startup and builds the study
form's factors from its stages and techniques.

To evaluate your own system, drop your techniques into that folder (or point
`CAFE_TECHNIQUES_DIR` at your own), then hit **Reload** on the Techniques page — no
restart needed. See [Define your system](define-a-system.md) for how to write the
pipeline.

## The flow

1. **Techniques** — the discovered pipeline: its stages, the techniques on each, and
   the factors they become. **Reload** picks up edits to the techniques folder.
2. **Datasets & Rubrics** — pick the inputs to evaluate on and the scale to grade with
   (the built-in rubrics, or your own).
3. **New study** — choose which stages/parameters to vary as factors, the questions, a
   judge and rubric, and the number of replications. CAFE shows the configuration count
   and a cost/time estimate before you commit.
4. **Run** — launch it and watch **live progress** stream as cells complete; long runs
   are resumable.
5. **Results** — the same analysis as `report()`, in the browser: per-factor
   attribution (F, p, partial η², variance share), effect sizes, per-level marginal
   means, and the best configuration. (See
   [Interpreting results](../interpreting-results.md) for how to read these.)
6. **Raters** — collect human ratings by name and see judge↔human reliability, the
   platform version of the [human-ratings workflow](human-ratings.md).

## Local development (without Docker)

```bash
# backend
cd apps/web-app/backend
pip install -r requirements.txt && pip install -e ../../../packages/cafe-core
uvicorn app.main:app --reload

# frontend (another shell)
cd apps/web-app/frontend && npm install && npm run dev
```

Requires Python ≥ 3.11 + R (for the stats layer) and Node for the front-end, plus a
reachable Postgres — the Docker path bundles all of these, which is why it is the
recommended way to run the app.
