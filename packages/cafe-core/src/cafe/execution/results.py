"""Observation + Results — what a study run produces.

An ``Observation`` is one executed cell: one (configuration x input x replication)
with its output, timing, cost, and any error. ``Results`` is the collection plus
light convenience (records, JSONL, a summary, optional DataFrame).

In library mode this object is simply *returned* from ``run_study`` — you hold it
in a variable, like normal code. The on-disk checkpoint (see
:mod:`cafe.checkpoint`) is separate plumbing for crash-safety/resume.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def config_id(config: dict[str, Any]) -> str:
    """Stable short id for a configuration (order-independent)."""
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def level_label(value: Any) -> str:
    """Display a factor level. A ``None`` level (a skipped stage) renders as ``none`` so
    it's a clean category everywhere (labels, DataFrames, the statistical model) rather
    than a missing/NaN value."""
    return "none" if value is None else value


def config_label(config: dict[str, Any]) -> str:
    """Human-readable one-line label, e.g. ``model=large·prompt=cot``."""
    return "·".join(f"{k}={level_label(config[k])}" for k in sorted(config))


@dataclass
class Observation:
    """One executed (config x input x replication) cell."""

    config: dict[str, Any]
    input_id: str
    rep: int
    output: Any = None
    error: str | None = None
    elapsed_s: float | None = None
    started_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def config_id(self) -> str:
        return config_id(self.config)

    @property
    def ok(self) -> bool:
        return self.error is None

    def key(self) -> str:
        """Idempotency key: (config, input, replication)."""
        return f"{self.config_id}::{self.input_id}::{self.rep}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Observation":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Results:
    """All observations from one study run, plus the factor list for context."""

    study_name: str
    factors: list[str]
    observations: list[Observation] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.observations)

    def __iter__(self):
        return iter(self.observations)

    @property
    def errors(self) -> list[Observation]:
        return [o for o in self.observations if not o.ok]

    def summary(self) -> dict[str, Any]:
        configs = {o.config_id for o in self.observations}
        inputs = {o.input_id for o in self.observations}
        oks = [o for o in self.observations if o.ok]
        timed = [o.elapsed_s for o in oks if o.elapsed_s is not None]
        return {
            "study": self.study_name,
            "n_answers": len(self.observations),
            "n_configs": len(configs),
            "n_inputs": len(inputs),
            "n_errors": len(self.errors),
            "total_compute_s": round(sum(timed), 4) if timed else 0.0,
            "mean_cell_s": round(sum(timed) / len(timed), 4) if timed else None,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"Results({self.study_name}: {s['n_answers']} answers, "
            f"{s['n_configs']} configs, {s['n_errors']} errors)"
        )

    @property
    def df(self):
        """A tidy pandas DataFrame of the answers (needs the ``stats`` extra)."""
        import pandas as pd

        return pd.DataFrame(self.to_records())

    def _repr_html_(self) -> str:  # pragma: no cover - notebook display
        try:
            df = self.df
            note = "" if len(df) <= 20 else f"<i>showing 20 of {len(df)} rows</i>"
            factors = ", ".join(self.factors) or "—"
            return (
                f"<b>{self!r}</b><br><small>factor columns: <b>{factors}</b></small>"
                f"{df.head(20).to_html(index=False)}{note}"
            )
        except Exception:
            return f"<b>{self!r}</b>"

    # Column names that the frame reserves for itself; a factor or metadata key
    # colliding with one of these is suffixed to keep both visible.
    _RESERVED = ("input_id", "rep", "output", "elapsed_s", "error", "config_id")

    def to_records(self) -> list[dict[str, Any]]:
        """Flat rows for a DataFrame/CSV: one column per factor (by its own name),
        then the answer and timing, then any metadata the system returned.

        Columns: ``<each factor>``, ``input_id``, ``rep``, ``output``,
        ``elapsed_s``, ``error``, ``<each metadata key>``.
        """
        rows: list[dict[str, Any]] = []
        for o in self.observations:
            row: dict[str, Any] = {}
            for name, value in o.config.items():  # factors first, by their own names
                row[name if name not in self._RESERVED else f"{name}_factor"] = level_label(value)
            row["input_id"] = o.input_id
            row["rep"] = o.rep
            row["output"] = o.output
            row["elapsed_s"] = o.elapsed_s
            row["error"] = o.error
            for key, value in o.metadata.items():
                row[key if key not in row else f"{key}_meta"] = value
            rows.append(row)
        return rows

    def to_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for o in self.observations:
                fh.write(json.dumps(o.to_dict(), default=str) + "\n")

    def to_df(self):  # pragma: no cover - thin pandas adapter
        """Return a pandas DataFrame (requires the ``stats`` extra)."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Results.to_df() needs pandas; install with: pip install 'cafe-core[stats]'"
            ) from exc
        return pd.DataFrame(self.to_records())
