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


@router.get("/pipelines/errors")
def pipeline_errors() -> dict[str, str]:
    """The files of the techniques folder that could not be imported, so have no pipeline.

    Returns
    -------
    dict
        ``{file name: error}`` from the last scan (``/pipelines``, ``/pipelines/reload``), e.g.
        ``{"rag.py": "ModuleNotFoundError: No module named 'faiss'"}``; ``{}`` when every file
        imported. The backend log has each one's traceback.
    """
    return pipeline_loader.import_errors()
