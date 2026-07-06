"""Runtime configuration, from environment variables (see .env.example)."""

from __future__ import annotations

import os
from pathlib import Path

# Where the user's technique pipeline lives (a folder exposing a `cafe.Pipeline` as `pipe`).
# In Docker this is set to /techniques (mounted). For local dev (no env var) we resolve the
# repo-root `techniques/` folder from this file's location — guarded so it never crashes import
# if the layout differs (e.g. inside the container the backend lives at /app).
_env_dir = os.environ.get("CAFE_TECHNIQUES_DIR")
if _env_dir:
    TECHNIQUES_DIR = Path(_env_dir)
else:
    _here = Path(__file__).resolve()
    # apps/web-app/backend/app/config.py → repo root is 4 levels up.
    _repo_root = _here.parents[4] if len(_here.parents) > 4 else _here.parent
    TECHNIQUES_DIR = _repo_root / "techniques"

# Database (async driver). docker-compose sets Postgres; local dev falls back to SQLite so the
# app runs with zero setup. Both work via SQLAlchemy async.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./cafe.db")

# CORS origin for the frontend dev server / container.
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")

# LLM credentials are read by cafe-core / litellm directly from the environment
# (OLLAMA_API_KEY, OPENROUTER_API_KEY, …). We only surface whether they're present.
def missing_llm_keys() -> list[str]:
    return [k for k in ("OLLAMA_API_KEY", "OPENROUTER_API_KEY") if not os.environ.get(k)]
