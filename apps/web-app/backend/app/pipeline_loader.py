"""Discover pipelines from CAFE_TECHNIQUES_DIR.

**Each `.py` file whose module exposes a `pipe` (a `cafe.Pipeline`) with a `@pipe.compose` is its own
selectable system**, keyed by the file's stem. A study binds to one of them. Files without a compose
are technique libraries you import into a compose file — not independently runnable. cafe-core is
untouched; this is web-app-only glue. `reload()` re-scans the folder for new/changed files.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from app.config import TECHNIQUES_DIR

_CACHE: dict[str, Any] = {"pipelines": None}


def _import(pyfile: Path):
    # Fresh spec each call → re-executes the file, so reload() picks up edits.
    spec = importlib.util.spec_from_file_location(f"cafe_user_{pyfile.stem}", pyfile)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def discover_pipelines(directory: Path) -> dict[str, Any]:
    """{name: cafe.Pipeline} for every `.py` file with a composed pipeline."""
    out: dict[str, Any] = {}
    for pyfile in sorted(p for p in directory.glob("*.py") if not p.name.startswith("_")):
        try:
            module = _import(pyfile)
        except Exception:  # noqa: BLE001 — a broken file shouldn't break discovery of the others
            continue
        pipe = getattr(module, "pipe", None)
        if pipe is not None and getattr(pipe, "_compose", None) is not None:
            out[pyfile.stem] = pipe
    return out


def get_pipelines(reload: bool = False) -> dict[str, Any]:
    if reload or _CACHE["pipelines"] is None:
        _CACHE["pipelines"] = discover_pipelines(TECHNIQUES_DIR)
    return _CACHE["pipelines"]


def get_pipeline(name: str):
    pipes = get_pipelines()
    if name not in pipes:
        raise KeyError(f"no pipeline named {name!r}; have: {sorted(pipes)}")
    return pipes[name]


def describe(pipe) -> list[dict[str, Any]]:
    """The stages/techniques/params of one pipeline (for the UI + factor builder)."""
    stages = []
    for stage in pipe.stages():
        techniques = []
        for tname in pipe.names_for(stage):
            spec = pipe.get(stage, tname)
            techniques.append({
                "name": tname, "description": spec.description, "cost_usd": spec.cost_usd,
                "params": [{"name": p, "default": d} for p, d in spec.params.items()],
            })
        stages.append({"stage": stage, "techniques": techniques})
    return stages


def list_pipelines(reload: bool = False) -> list[dict[str, Any]]:
    return [{"name": name, "stages": describe(pipe)} for name, pipe in get_pipelines(reload).items()]
