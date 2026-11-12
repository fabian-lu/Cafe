#!/usr/bin/env python3
"""Export studies from a running CAFE web app as the static snapshot behind the read-only demo.

    python scripts/export_demo_data.py --studies 17 18 19 --pipelines question_answering support_chatbot

Writes apps/web-app/frontend/public/demo-data/ (or the directory given with ``--out``): every GET
response the frontend needs for these studies, under the file names lib/demo.js's demoFile()
expects. Each results view is its own file: every dimension, and every value of every question
filter (those are recomputed by the backend, so exporting a study with many filter values takes a
few minutes).

The new snapshot is written to a temporary directory first and replaces the old one only once every
request has succeeded: if the backend isn't running or ``--api`` is wrong, the old snapshot stays.

To keep the files small, each results view stores every question text and reference answer once
(``questions`` / ``references``, keyed by question id) instead of in every answer row; the Results
page fills them back in.
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "apps" / "web-app" / "frontend" / "public" / "demo-data"
KEEP = {"judge-preview.json"}   # a POST response that isn't tied to a study; kept as it is


class ExportError(Exception):
    """The backend didn't give the export something it needs; the old snapshot is left as it is."""


def file_name(path: str, params: dict | None = None) -> str:
    """The demo file for an API path; mirrors demoFile() in apps/web-app/frontend/src/lib/demo.js."""
    name = path.strip("/").replace("/", "-")
    view = [f"{key}-{params[key]}" for key in ("dimension", "fkey", "fval") if params and params.get(key)]
    if view:
        name += "--" + re.sub(r"[^A-Za-z0-9_.-]", "_", "--".join(view))
    return name + ".json"


def fetch(api: str, path: str, params: dict | None = None, *, required: bool = False):
    """GET an API path and decode its JSON.

    Parameters
    ----------
    api : str
        The backend's API URL, e.g. ``http://localhost:8000/api``.
    path : str
        The API path, e.g. ``/studies/17/results``.
    params : dict, optional
        The query parameters.
    required : bool, default False
        Whether the snapshot can't do without this response. If it can, an HTTP error (the backend
        has no such view, e.g. no human ratings yet) returns None and the demo shows an empty view.

    Returns
    -------
    object or None
        The decoded JSON, or None for an HTTP error on a path that isn't required.

    Raises
    ------
    ExportError
        If the backend can't be reached or doesn't answer with JSON, or for an HTTP error on a
        required path.
    """
    url = api + path + ("?" + urllib.parse.urlencode(params) if params else "")
    try:
        with urllib.request.urlopen(url, timeout=900) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if required:
            raise ExportError(f"{url} answered HTTP {exc.code} {exc.reason}; is --api the "
                              "backend's API URL (e.g. http://localhost:8000/api)?") from exc
        return None
    except urllib.error.URLError as exc:
        raise ExportError(f"can't reach the backend at {url} ({exc.reason}); is it running, and "
                          "is --api right?") from exc
    except json.JSONDecodeError as exc:
        raise ExportError(f"{url} didn't answer with JSON; is --api the backend's "
                          "API URL?") from exc
    # e.g. a timeout or a dropped connection
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise ExportError(f"GET {url} failed: {exc!r}") from exc


def save(out: Path, name: str, data) -> None:
    """Write one file of the snapshot.

    Parameters
    ----------
    out : Path
        The directory to write to.
    name : str
        The file name, from file_name().
    data : object
        The JSON-serialisable content.
    """
    (out / name).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def compact(results: dict) -> dict:
    """Store each question and reference once per view instead of in every answer row."""
    questions, references = {}, {}
    for row in results.get("records", []):
        questions[row["input_id"]] = row.pop("question", None)
        references[row["input_id"]] = row.pop("reference", None)
    return {**results, "questions": questions, "references": references}


def export(api: str, study_ids: list[int], pipeline_names: list[str] | None, out: Path) -> None:
    """Write the snapshot of these studies, with the datasets, rubrics and pipelines around them.

    Parameters
    ----------
    api : str
        The backend's API URL.
    study_ids : list of int
        The studies to export.
    pipeline_names : list of str, optional
        Only these pipelines on the Techniques page; all of them if None or empty.
    out : Path
        The (empty) directory to write the files to.

    Raises
    ------
    ExportError
        If the backend can't be reached, ``api`` isn't its API URL, or a study isn't there.
    """
    save(out, "health.json", fetch(api, "/health", required=True))
    save(out, "judge-presets.json", fetch(api, "/judge/presets"))
    pipelines = fetch(api, "/pipelines", required=True)
    if pipeline_names:
        pipelines = [p for p in pipelines if p.get("name") in pipeline_names]
    save(out, "pipelines.json", pipelines)

    studies = [s for s in fetch(api, "/studies", required=True) if s["id"] in study_ids]
    if missing := sorted(set(study_ids) - {s["id"] for s in studies}):
        raise ExportError(f"no study {', '.join(map(str, missing))} at {api} "
                          "(archived studies aren't listed)")
    save(out, "studies.json", studies)
    dataset_ids, rubric_ids = set(), set()

    for study in studies:
        sid = study["id"]
        dataset_ids.add(study["dataset_id"])
        rubric_ids.update(study.get("rubric_ids") or [study["rubric_id"]])
        save(out, file_name(f"/studies/{sid}"), fetch(api, f"/studies/{sid}"))

        dimensions = fetch(api, f"/studies/{sid}/dimensions") or []
        filters = fetch(api, f"/studies/{sid}/filters") or {}
        save(out, file_name(f"/studies/{sid}/dimensions"), dimensions)
        save(out, file_name(f"/studies/{sid}/filters"), filters)

        # Without a dimension, the backend answers with the primary dimension's results: the study
        # page asks for this view (for the run time) whatever the study's dimensions.
        results = fetch(api, f"/studies/{sid}/results")
        if results is not None:   # None: the study has no results yet
            save(out, file_name(f"/studies/{sid}/results"), compact(results))
        for dimension in [d["dimension"] for d in dimensions]:
            views = [{"dimension": dimension}] + [{"dimension": dimension, "fkey": key, "fval": value}
                                                  for key, values in filters.items() for value in values]
            for params in views:
                results = fetch(api, f"/studies/{sid}/results", params)
                if results is not None:
                    save(out, file_name(f"/studies/{sid}/results", params), compact(results))
            print(f"  {study['name']}: {dimension}, {len(views)} views", flush=True)

        for extra in ("raters", "reliability", "rating-sheet"):
            data = fetch(api, f"/studies/{sid}/{extra}")
            if data is not None:
                save(out, file_name(f"/studies/{sid}/{extra}"), data)

    datasets = fetch(api, "/datasets", required=True)
    save(out, "datasets.json", [d for d in datasets if d["id"] in dataset_ids])
    for did in dataset_ids:
        save(out, file_name(f"/datasets/{did}"), fetch(api, f"/datasets/{did}"))
    rubrics = fetch(api, "/rubrics", required=True)
    save(out, "rubrics.json", [r for r in rubrics if r["id"] in rubric_ids])
    for rid in rubric_ids:
        save(out, file_name(f"/rubrics/{rid}"), fetch(api, f"/rubrics/{rid}"))


def replace_snapshot(new: Path, out: Path) -> None:
    """Replace the snapshot in ``out`` with a finished export.

    The new files are moved in first (a rename replaces a file in one step), then the old
    snapshot's remaining ``.json`` files are deleted, except the ones in KEEP.

    Parameters
    ----------
    new : Path
        The directory the export was written to, on the same file system as ``out``.
    out : Path
        The snapshot directory; created if it doesn't exist yet.
    """
    out.mkdir(exist_ok=True)
    written = {path.name for path in new.glob("*.json")}
    for name in written:
        (new / name).replace(out / name)
    for old in out.glob("*.json"):
        if old.name not in written and old.name not in KEEP:
            old.unlink()


def main() -> None:
    """Export into a temporary directory, then let the export replace the snapshot.

    Raises
    ------
    SystemExit
        If the export fails; the snapshot is then left as it was.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--api", default="http://localhost:8000/api")
    parser.add_argument("--studies", type=int, nargs="+", required=True)
    parser.add_argument("--pipelines", nargs="*", help="only these pipelines on the Techniques page")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="the snapshot directory (default: the demo's, %(default)s)")
    args = parser.parse_args()

    out = args.out.resolve()
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        # next to the snapshot, so its files can replace the old ones once everything is there
        with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as new:
            export(args.api, args.studies, args.pipelines, Path(new))
            replace_snapshot(Path(new), out)
    except ExportError as exc:
        sys.exit(f"export failed, {shown} is unchanged: {exc}")

    total = sum(f.stat().st_size for f in out.glob("*.json"))
    print(f"wrote {len(list(out.glob('*.json')))} files, {total / 1e6:.1f} MB, to {shown}")


if __name__ == "__main__":
    main()
