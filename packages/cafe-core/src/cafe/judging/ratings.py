"""Verdict data types: a single :class:`JudgeOutput`/:class:`Rating` and the
:class:`Ratings` collection (with a tidy DataFrame view)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cafe.judging.rubric import Rubric


@dataclass
class JudgeOutput:
    """One judge call's result, including the exact prompt and raw response."""

    value: Any
    value_numeric: int | None
    reasoning: str | None
    prompt: str
    raw_response: str | None


@dataclass
class Rating:
    """One judge verdict for one observation.

    ``prompt`` and ``raw_response`` record exactly what was sent and what came back
    (before parsing) so every verdict is auditable. ``config`` is denormalized onto
    the rating for easy statistics.
    """

    obs_key: str
    config: dict[str, Any]
    input_id: str
    rep: int
    judge_rep: int
    value: Any = None
    value_numeric: int | None = None
    reasoning: str | None = None
    error: str | None = None
    prompt: str | None = None
    raw_response: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.value_numeric is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Ratings:
    """All judge verdicts for a study, plus the rubric they were scored on."""

    rubric: Rubric
    judge_model: str
    factors: list[str] = field(default_factory=list)
    items: list[Rating] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    @property
    def errors(self) -> list[Rating]:
        return [r for r in self.items if not r.ok]

    _RESERVED = ("input_id", "rep", "judge_rep", "verdict", "reasoning", "error")

    def to_records(self) -> list[dict[str, Any]]:
        """Flat rows: one column per factor (by name), then verdict + reasoning."""
        rows = []
        for r in self.items:
            row: dict[str, Any] = {}
            for name, value in r.config.items():
                row[name if name not in self._RESERVED else f"{name}_factor"] = value
            row["input_id"] = r.input_id
            row["rep"] = r.rep
            row["judge_rep"] = r.judge_rep
            row["verdict"] = r.value_numeric
            row["reasoning"] = r.reasoning
            row["error"] = r.error
            rows.append(row)
        return rows

    @property
    def df(self):
        """A pandas DataFrame of the ratings (needs the ``stats`` extra)."""
        import pandas as pd

        return pd.DataFrame(self.to_records())

    def to_df(self):  # backwards-friendly alias
        return self.df

    def _repr_html_(self) -> str:  # pragma: no cover - notebook display
        try:
            view = self.df.drop(columns=["error"], errors="ignore")
            note = "" if len(view) <= 20 else f"<i>showing 20 of {len(view)} rows</i>"
            return (
                f"<b>Ratings</b> — {len(self)} verdicts on "
                f"<code>{self.rubric.name}</code>, judge <code>{self.judge_model}</code>"
                f"{view.head(20).to_html(index=False)}{note}"
            )
        except Exception:
            return f"<b>Ratings</b> ({len(self)} verdicts)"
