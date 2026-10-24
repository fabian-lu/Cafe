#!/usr/bin/env python3
"""Export studies from a running CAFE web app as the static snapshot behind the read-only demo.

    python scripts/export_demo_data.py --studies 17 18 19 --pipelines question_answering support_chatbot

Writes apps/web-app/frontend/public/demo-data/: every GET response the frontend needs for these
studies, under the file names lib/demo.js's demoFile() expects. Each results view is its own file:
every dimension, and every value of every question filter (those are recomputed by the backend, so
exporting a study with many filter values takes a few minutes).

To keep the files small, each results view stores every question text and reference answer once
(``questions`` / ``references``, keyed by question id) instead of in every answer row; the Results
page fills them back in.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "apps" / "web-app" / "frontend" / "public" / "demo-data"
KEEP = {"judge-preview.json"}   # a POST response that isn't tied to a study; kept as it is


def file_name(path: str, params: dict | None = None) -> str:
    """The demo file for an API path; mirrors demoFile() in apps/web-app/frontend/src/lib/demo.js."""
    name = path.strip("/").replace("/", "-")
    view = [f"{key}-{params[key]}" for key in ("dimension", "fkey", "fval") if params and params.get(key)]
    if view:
        name += "--" + re.sub(r"[^A-Za-z0-9_.-]", "_", "--".join(view))
    return name + ".json"


def fetch(api: str, path: str, params: dict | None = None):
    url = api + path + ("?" + urllib.parse.urlencode(params) if params else "")
    try:
        with urllib.request.urlopen(url, timeout=900) as response:
            return json.load(response)
    except urllib.error.HTTPError:
        return None   # e.g. no human ratings yet: the demo falls back to an empty view


def save(name: str, data) -> None:
    (OUT / name).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def compact(results: dict) -> dict:
    """Store each question and reference once per view instead of in every answer row."""
    questions, references = {}, {}
    for row in results.get("records", []):
        questions[row["input_id"]] = row.pop("question", None)
        references[row["input_id"]] = row.pop("reference", None)
    return {**results, "questions": questions, "references": references}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--api", default="http://localhost:8000/api")
    parser.add_argument("--studies", type=int, nargs="+", required=True)
    parser.add_argument("--pipelines", nargs="*", help="only these pipelines on the Techniques page")
    args = parser.parse_args()

    for old in OUT.glob("*.json"):
        if old.name not in KEEP:
            old.unlink()

    save("health.json", fetch(args.api, "/health"))
    save("judge-presets.json", fetch(args.api, "/judge/presets"))
    pipelines = fetch(args.api, "/pipelines")
    if args.pipelines:
        pipelines = [p for p in pipelines if p.get("name") in args.pipelines]
    save("pipelines.json", pipelines)

    studies = [s for s in fetch(args.api, "/studies") if s["id"] in args.studies]
    save("studies.json", studies)
    dataset_ids, rubric_ids = set(), set()

    for study in studies:
        sid = study["id"]
        dataset_ids.add(study["dataset_id"])
        rubric_ids.update(study.get("rubric_ids") or [study["rubric_id"]])
        save(file_name(f"/studies/{sid}"), fetch(args.api, f"/studies/{sid}"))

        dimensions = fetch(args.api, f"/studies/{sid}/dimensions") or []
        filters = fetch(args.api, f"/studies/{sid}/filters") or {}
        save(file_name(f"/studies/{sid}/dimensions"), dimensions)
        save(file_name(f"/studies/{sid}/filters"), filters)

        if not dimensions:   # the frontend only asks without a dimension when a study has none
            save(file_name(f"/studies/{sid}/results"), compact(fetch(args.api, f"/studies/{sid}/results")))
        for dimension in [d["dimension"] for d in dimensions]:
            views = [{"dimension": dimension}] + [{"dimension": dimension, "fkey": key, "fval": value}
                                                  for key, values in filters.items() for value in values]
            for params in views:
                results = fetch(args.api, f"/studies/{sid}/results", params)
                if results is not None:
                    save(file_name(f"/studies/{sid}/results", params), compact(results))
            print(f"  {study['name']}: {dimension}, {len(views)} views", flush=True)

        for extra in ("raters", "reliability", "rating-sheet"):
            data = fetch(args.api, f"/studies/{sid}/{extra}")
            if data is not None:
                save(file_name(f"/studies/{sid}/{extra}"), data)

    save("datasets.json", [d for d in fetch(args.api, "/datasets") if d["id"] in dataset_ids])
    for did in dataset_ids:
        save(file_name(f"/datasets/{did}"), fetch(args.api, f"/datasets/{did}"))
    save("rubrics.json", [r for r in fetch(args.api, "/rubrics") if r["id"] in rubric_ids])
    for rid in rubric_ids:
        save(file_name(f"/rubrics/{rid}"), fetch(args.api, f"/rubrics/{rid}"))

    total = sum(f.stat().st_size for f in OUT.glob("*.json"))
    print(f"wrote {len(list(OUT.glob('*.json')))} files, {total / 1e6:.1f} MB, to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
