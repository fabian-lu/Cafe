"""Discover pipelines from CAFE_TECHNIQUES_DIR.

**Each `.py` file whose module exposes a `pipe` (a `cafe.Pipeline`) with a `@pipe.compose` is its own
selectable system**, keyed by the file's stem. A study binds to one of them. Files without a compose
are technique libraries you import into a compose file — not independently runnable. cafe-core is
untouched; this is web-app-only glue. `reload()` re-scans the folder for new/changed files.

A file is imported the way `python <file>` runs it, so a compose file can `import` the technique
libraries next to it. A file that can't be imported is skipped, and :func:`import_errors` says why.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

from app.config import TECHNIQUES_DIR

log = logging.getLogger(__name__)

_CACHE: dict[str, Any] = {"pipelines": None, "errors": {}}
# A scan swaps the folder's modules in and out of sys.modules: one scan at a time.
_SCAN_LOCK = threading.Lock()


def _import(pyfile: Path):
    """Import one file of the techniques folder as module ``cafe_user_<stem>``.

    Parameters
    ----------
    pyfile : Path
        The file. It is executed afresh on every call, so a re-scan picks up edits.

    Returns
    -------
    module
        The module, registered in ``sys.modules`` like any imported module (``@dataclass``
        looks a class's module up there).
    """
    spec = importlib.util.spec_from_file_location(f"cafe_user_{pyfile.stem}", pyfile)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _forget_modules_from(directory: Path) -> None:
    """Drop the modules imported from a folder from ``sys.modules``, so that the next import
    reads their files again.

    Parameters
    ----------
    directory : Path
        The folder, resolved. A module counts if its file is in the folder, or if it belongs to
        a package in the folder; modules from further down (e.g. a virtualenv kept in the
        folder) and the script the server was started with (``__main__``) are left alone.
    """
    from_folder = {"__main__": False}  # top-level module name -> imported from the folder?
    for name in list(sys.modules):
        top = name.partition(".")[0]
        if top not in from_folder:
            try:
                path = Path(sys.modules[top].__file__).resolve()
            except Exception:  # noqa: BLE001 — no file: a built-in or namespace package, ...
                from_folder[top] = False
            else:
                from_folder[top] = path.parent == directory or (
                    path.name == "__init__.py" and path.parent.parent == directory)
        if from_folder[top]:
            sys.modules.pop(name, None)


def discover_pipelines(directory: Path, errors: dict[str, str] | None = None) -> dict[str, Any]:
    """The pipelines in a techniques folder: one per `.py` file with a composed pipeline.

    Each file is imported the way ``python <file>`` runs it: with the folder first on
    ``sys.path``, so a compose file can import the technique libraries next to it
    (``from retrieval import dense``), and registered in ``sys.modules``. The folder stays on
    ``sys.path`` afterwards, as a technique may also import a library while a study runs. The
    modules imported from the folder before are dropped first, so that a library edited since
    is read again.

    Parameters
    ----------
    directory : Path
        The techniques folder.
    errors : dict, optional
        Filled with ``{file name: error}`` for each file that could not be imported (a syntax
        error, a missing package, ...), e.g. ``{"rag.py": "ModuleNotFoundError: No module
        named 'faiss'"}``. Such a file is skipped and its traceback logged; the others are
        still discovered.

    Returns
    -------
    dict
        ``{name: cafe.Pipeline}``, keyed by the file's stem.
    """
    directory = directory.resolve()
    out: dict[str, Any] = {}
    with _SCAN_LOCK:
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        _forget_modules_from(directory)
        importlib.invalidate_caches()  # find libraries added since the last scan
        for pyfile in sorted(p for p in directory.glob("*.py") if not p.name.startswith("_")):
            try:
                module = _import(pyfile)
            except Exception as exc:  # noqa: BLE001 — a broken file shouldn't break discovery of the others
                log.warning("skipped %s: it could not be imported", pyfile, exc_info=True)
                if errors is not None:
                    errors[pyfile.name] = "".join(traceback.format_exception_only(exc)).strip()
                continue
            pipe = getattr(module, "pipe", None)
            if pipe is not None and getattr(pipe, "_compose", None) is not None:
                out[pyfile.stem] = pipe
    return out


def get_pipelines(reload: bool = False) -> dict[str, Any]:
    """The pipelines of the techniques folder.

    Parameters
    ----------
    reload : bool, optional
        Scan the folder again instead of returning the last scan's pipelines (the first call
        scans it either way). A scan also records which files could not be imported, see
        :func:`import_errors`.

    Returns
    -------
    dict
        ``{name: cafe.Pipeline}``, see :func:`discover_pipelines`.
    """
    if reload or _CACHE["pipelines"] is None:
        errors: dict[str, str] = {}
        pipes = discover_pipelines(TECHNIQUES_DIR, errors)
        _CACHE.update(pipelines=pipes, errors=errors)
    return _CACHE["pipelines"]


def import_errors() -> dict[str, str]:
    """The files of the techniques folder that could not be imported, so have no pipeline.

    Returns
    -------
    dict
        ``{file name: error}`` from the last scan (see :func:`discover_pipelines`); empty when
        every file imported. Scans the folder first if it hasn't been scanned yet.
    """
    get_pipelines()
    return dict(_CACHE["errors"])


def get_pipeline(name: str):
    """One discovered pipeline.

    Parameters
    ----------
    name : str
        The pipeline's name (its file's stem).

    Returns
    -------
    cafe.Pipeline
        The pipeline.

    Raises
    ------
    KeyError
        If there is no such pipeline; the message names the ones there are, and the import
        error if ``<name>.py`` could not be imported.
    """
    pipes = get_pipelines()
    if name not in pipes:
        why = _CACHE["errors"].get(f"{name}.py")
        raise KeyError(f"no pipeline named {name!r}; have: {sorted(pipes)}"
                       + (f" ({name}.py could not be imported: {why})" if why else ""))
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
