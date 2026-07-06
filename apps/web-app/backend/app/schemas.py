"""Pydantic request/response schemas (mirror cafe-core shapes)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class _ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ── Rubric ───────────────────────────────────────────────────────────────────
class RubricIn(BaseModel):
    name: str
    scale_type: str = "ordinal"
    levels: list[dict[str, Any]] = []          # [{value,label,description}]
    instruction: str = ""
    preset: str = "reference_qa"               # cafe judge preset (used when no custom template)
    system_prompt: str | None = None
    prompt_template: str | None = None         # custom, overrides preset


class RubricOut(_ORM, RubricIn):
    id: int


# ── Dataset (questions) ──────────────────────────────────────────────────────
class DatasetIn(BaseModel):
    name: str
    items: list[dict[str, Any]] = []           # [{id?,text,reference?}]


class DatasetOut(_ORM, DatasetIn):
    id: int


# ── Study ────────────────────────────────────────────────────────────────────
class StudyIn(BaseModel):
    name: str
    description: str = ""                        # optional free-text notes
    pipeline: str = "pipeline"                  # which discovered system (file stem)
    factors: list[dict[str, Any]] = []         # [{name, levels:[...]}]
    dataset_id: int | None = None
    rubric_id: int | None = None
    judge_model: str = ""
    replications: int = 1
    concurrency: int = 8


class StudyOut(_ORM, StudyIn):
    id: int
    status: str
    progress: int
    archived: bool = False
    created_at: datetime | None = None


class JudgePreviewIn(BaseModel):
    rubric: RubricIn
    judge_model: str = ""
    system_prompt: str | None = None
    question: str = "What is the capital of France?"
    answer: str = "Paris."
    reference: str | None = "Paris is the capital of France."
