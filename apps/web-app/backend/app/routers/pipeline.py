"""Pipeline discovery endpoints — the systems a study can run on."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app import pipeline_loader

router = APIRouter(prefix="/api", tags=["pipeline"])


@router.get("/pipelines")
def list_pipelines(reload: bool = False):
    """All discovered pipelines (each = a file with a @pipe.compose). Pass ?reload=true to re-scan."""
    try:
        return pipeline_loader.list_pipelines(reload=reload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"could not load pipelines: {exc}") from exc


@router.post("/pipelines/reload")
def reload_pipelines():
    """Re-scan the techniques folder for new/changed files."""
    return pipeline_loader.list_pipelines(reload=True)
