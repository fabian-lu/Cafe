# CAFE Platform (web app)

The interactive web platform that wraps `cafe-core`: define factorial studies over your compound-AI
system, launch them, watch progress, explore results, collect human ratings. See
[`design/05-web-app.md`](../../design/05-web-app.md) for the full architecture.

## Run it (Docker)

```bash
cd apps/web-app
cp .env.example .env        # add your OLLAMA_API_KEY / OPENROUTER_API_KEY
docker compose up
```

- Frontend → http://localhost:5173
- Backend API → http://localhost:8000/api/health, http://localhost:8000/api/pipeline

## The system under test

Lives in code, in the repo-root [`techniques/`](../../techniques) folder (mounted into the backend).
It must expose a module-level `pipe` (a `cafe.Pipeline`) in `pipeline.py`. The app discovers it at
startup and builds the UI's factors from its stages/techniques. Edit that folder (or point
`CAFE_TECHNIQUES_DIR` / the docker volume at your own) and restart the backend to pick up changes.

## Local dev (without Docker)

```bash
# backend
cd apps/web-app/backend && pip install -r requirements.txt && pip install -e ../../../packages/cafe-core
uvicorn app.main:app --reload
# frontend (another shell)
cd apps/web-app/frontend && npm install && npm run dev
```

## Status

Scaffold + discovery live (Techniques page reads the real pipeline). Studies / Questions / Rubrics /
Results / Raters are stubs — built in later slices (`design/05-web-app.md` §11).
