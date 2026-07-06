"""Database models. Lean by design — no auth/org tables (see design/05-web-app.md §5).

JSON columns hold the shapes cafe-core already uses (factor levels, rubric levels, per-answer
metadata), so the API mirrors the library without a rigid relational explosion.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Rubric(Base):
    """The full grading spec: the scale + how the judge is prompted (everything except the judge
    *model*, which is chosen per-study)."""
    __tablename__ = "rubrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    scale_type: Mapped[str] = mapped_column(String(20), default="ordinal")  # ordinal|numeric|binary
    levels: Mapped[list] = mapped_column(JSON, default=list)                # [{value,label,description}]
    instruction: Mapped[str] = mapped_column(Text, default="")
    preset: Mapped[str] = mapped_column(String(40), default="reference_qa")  # cafe judge preset
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)   # judge system message
    prompt_template: Mapped[str | None] = mapped_column(Text, nullable=True)  # custom, overrides preset
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Dataset(Base):
    """A named set of questions (inputs). Each item: {id, text, reference?, metadata?}."""
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    items: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Study(Base):
    __tablename__ = "studies"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")       # optional free-text notes
    pipeline: Mapped[str] = mapped_column(String(200), default="pipeline")  # which discovered system
    factors: Mapped[list] = mapped_column(JSON, default=list)       # [{name, levels:[...]}]
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    rubric_id: Mapped[int | None] = mapped_column(ForeignKey("rubrics.id"), nullable=True)
    judge_model: Mapped[str] = mapped_column(String(200), default="")
    replications: Mapped[int] = mapped_column(Integer, default=1)
    concurrency: Mapped[int] = mapped_column(Integer, default=8)       # parallel calls per phase
    status: Mapped[str] = mapped_column(String(20), default="draft")  # draft|running|done|failed
    progress: Mapped[int] = mapped_column(Integer, default=0)         # 0..100
    archived: Mapped[bool] = mapped_column(Boolean, default=False)    # soft-delete: hidden, restorable
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    answers: Mapped[list["Answer"]] = relationship(back_populates="study")


class Answer(Base):
    __tablename__ = "answers"
    id: Mapped[int] = mapped_column(primary_key=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.id"))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    input_id: Mapped[str] = mapped_column(String(200))
    rep: Mapped[int] = mapped_column(Integer, default=0)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    study: Mapped[Study] = relationship(back_populates="answers")
    ratings: Mapped[list["Rating"]] = relationship(back_populates="answer", cascade="all, delete")


class Rating(Base):
    """A judge verdict for an answer."""
    __tablename__ = "ratings"
    id: Mapped[int] = mapped_column(primary_key=True)
    answer_id: Mapped[int] = mapped_column(ForeignKey("answers.id"))
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    answer: Mapped[Answer] = relationship(back_populates="ratings")


class HumanRating(Base):
    """A human rater's score for an answer (rater identity is just a name — no accounts)."""
    __tablename__ = "human_ratings"
    id: Mapped[int] = mapped_column(primary_key=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.id"))
    answer_key: Mapped[str] = mapped_column(String(300))   # cafe obs key (config·input·rep)
    rater: Mapped[str] = mapped_column(String(120))
    score: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class StudyResult(Base):
    """Cached, JSON-serialised results for a finished study (report/effects/marginals/pareto/records).
    Computed once when the run completes; the Results page reads this blob."""
    __tablename__ = "study_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.id"), unique=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
