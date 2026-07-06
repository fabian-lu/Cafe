#!/usr/bin/env python3
"""Copy the tutorial notebooks from examples/ into docs/notebooks/ for the docs build.

examples/ is the single source of truth for the notebooks; docs/notebooks/ is generated (gitignored)
so the two can't drift. Run before `mkdocs build` / `mkdocs serve` — CI does this automatically.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "examples"
DST = ROOT / "docs" / "notebooks"

# Notebooks referenced by mkdocs.yml's nav (relative to examples/ → docs/notebooks/).
NOTEBOOKS = [
    "01_quickstart.ipynb",
    "02_technique_mode.ipynb",
    "03_human_and_irr.ipynb",
    "04_judging_modes.ipynb",
    "05_cost_quality.ipynb",
    "06_fractional_design.ipynb",
    "evaluation/evaluation.ipynb",
]


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    for rel in NOTEBOOKS:
        src = SRC / rel
        if not src.exists():
            raise SystemExit(f"missing notebook: {src}")
        dst = DST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    print(f"synced {len(NOTEBOOKS)} notebooks into {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
