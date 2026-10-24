#!/usr/bin/env python3
"""Copy the notebooks from examples/ and case_studies/ into docs/notebooks/ for the docs build.

examples/ and case_studies/ are the single source of truth for the notebooks; docs/notebooks/ is
generated (gitignored) so the two can't drift. Run before `mkdocs build` / `mkdocs serve` — CI does
this automatically.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "docs" / "notebooks"

# Notebooks referenced by mkdocs.yml's nav: (source relative to the repo root, destination relative
# to docs/notebooks/).
NOTEBOOKS = [
    ("examples/01_quickstart.ipynb", "01_quickstart.ipynb"),
    ("examples/02_technique_mode.ipynb", "02_technique_mode.ipynb"),
    ("examples/03_human_and_irr.ipynb", "03_human_and_irr.ipynb"),
    ("examples/04_judging_modes.ipynb", "04_judging_modes.ipynb"),
    ("examples/05_cost_quality.ipynb", "05_cost_quality.ipynb"),
    ("examples/06_fractional_design.ipynb", "06_fractional_design.ipynb"),
    ("examples/evaluation/evaluation.ipynb", "evaluation/evaluation.ipynb"),
    ("case_studies/1_question_answering/question_answering_study.ipynb",
     "case_studies/question_answering_study.ipynb"),
    ("case_studies/2_support_chatbot/support_chatbot_study.ipynb", "case_studies/support_chatbot_study.ipynb"),
    ("case_studies/3_invoice_processing/invoice_study.ipynb", "case_studies/invoice_study.ipynb"),
]


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    for source, destination in NOTEBOOKS:
        src = ROOT / source
        if not src.exists():
            raise SystemExit(f"missing notebook: {src}")
        dst = DST / destination
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    print(f"synced {len(NOTEBOOKS)} notebooks into {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
