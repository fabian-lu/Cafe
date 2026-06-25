"""Tiny zero-dependency .env loader.

Looks for the nearest ``.env`` file (current directory upward) and loads its
``KEY=VALUE`` lines into ``os.environ`` without overwriting anything already set.
Kept dependency-free on purpose — the engine never requires python-dotenv.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(start: str | None = None) -> str | None:
    """Load the nearest ``.env`` into the environment. Returns the path used, or None.

    Existing environment variables win (we only ``setdefault``), so real env vars
    and CI secrets always override a local ``.env``.
    """
    base = Path(start or os.getcwd()).resolve()
    for cur in [base, *base.parents]:
        candidate = cur / ".env"
        if candidate.is_file():
            for raw in candidate.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), value)
            return str(candidate)
    return None
