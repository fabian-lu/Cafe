"""Judge presets + a live judge-prompt preview (build the LLMJudge + Rubric, show the exact prompt)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

import cafe
from cafe.judging.prompts import JUDGE_PRESETS

from app import schemas

router = APIRouter(prefix="/api", tags=["judge"])


@router.get("/judge/presets")
def presets():
    """The judge prompt presets cafe ships (name + template) for the Rubrics dropdown."""
    return [{"name": name, "template": tmpl} for name, tmpl in JUDGE_PRESETS.items()]


@router.post("/judge/preview")
def preview(body: schemas.JudgePreviewIn):
    """Return the exact [SYSTEM]/[USER] the judge would see for the given rubric — no LLM call."""
    try:
        r = body.rubric
        rubric = cafe.Rubric(
            name=r.name or "rubric", scale_type=r.scale_type,
            levels=[cafe.Level(int(l["value"]), l.get("label", ""), l.get("description", ""))
                    for l in r.levels],
            instruction=r.instruction or "Judge the answer.",
            prompt_template=r.prompt_template or None,
        )
        judge = cafe.LLMJudge(
            model=body.judge_model or "preview",
            preset=r.preset or "reference_qa",
            system_prompt=body.system_prompt or r.system_prompt or None,
        )
        return {"preview": judge.preview(rubric, body.question, body.answer, reference=body.reference)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"could not build preview: {exc}") from exc
