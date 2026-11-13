"""Pydantic request/response schemas (mirror cafe-core shapes)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict


class _ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


def _as_utc(value: datetime) -> datetime:
    """A stored timestamp as an aware UTC datetime, so the API sends it with an offset.

    The app stores UTC (``models._now``), but SQLite keeps no time zone: there a
    ``DateTime(timezone=True)`` column comes back naive, and was sent without an offset,
    which a browser reads as local time. Postgres returns it aware.

    Parameters
    ----------
    value : datetime
        The timestamp as the database returned it; a naive one is taken to be UTC.

    Returns
    -------
    datetime
        The same instant in UTC, serialized as e.g. ``"2026-11-12T20:31:29.514952Z"``.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


#: A timestamp in an API response: always UTC, with an explicit offset (see :func:`_as_utc`).
UTCDateTime = Annotated[datetime, AfterValidator(_as_utc)]


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
    rubric_id: int | None = None               # primary dimension
    rubric_ids: list[int] = []                 # ALL dimensions incl. primary ([] = just rubric_id)
    judge_model: str = ""
    replications: int = 1
    concurrency: int = 8


class StudyOut(_ORM, StudyIn):
    id: int
    status: str
    progress: int
    archived: bool = False
    created_at: UTCDateTime | None = None


class JudgePreviewIn(BaseModel):
    rubric: RubricIn
    judge_model: str = ""
    system_prompt: str | None = None
    question: str = "What is the capital of France?"
    answer: str = "Paris."
    reference: str | None = "Paris is the capital of France."
