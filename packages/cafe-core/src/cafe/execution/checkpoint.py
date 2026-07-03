"""Resumable checkpoint: append-only JSONL of observations.

Long studies run for hours; a crash must not lose everything. Every completed
observation is appended immediately. On resume, already-done cells (keyed by
config x input x replication) are skipped, so re-running fills only what's
missing. This is the library-mode equivalent of the platform worker's
Postgres-backed idempotency.

The file is plumbing: callers get their data from the returned ``Results``
object; the checkpoint just makes the run crash-safe.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Iterator

from cafe.execution.results import Observation

if TYPE_CHECKING:
    from cafe.judging.ratings import Rating


class Checkpoint:
    """Append-only JSONL store of observations, keyed for idempotent resume."""

    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

    def load(self) -> dict[str, Observation]:
        """Return existing observations keyed by their idempotency key (last wins)."""
        done: dict[str, Observation] = {}
        if not os.path.exists(self.path):
            return done
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obs = Observation.from_dict(json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    # Tolerate a torn final line from a hard crash mid-write.
                    continue
                done[obs.key()] = obs
        return done

    def append(self, obs: Observation) -> None:
        """Append one observation, flushing to disk so a crash loses at most this cell."""
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obs.to_dict(), default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def __iter__(self) -> Iterator[Observation]:
        return iter(self.load().values())


class RatingsCheckpoint:
    """Append-only JSONL store of judge verdicts, keyed by ``(obs_key, judge_rep)``.

    The judging phase is often the expensive one; this makes it crash-safe the same way
    :class:`Checkpoint` does for answers — each verdict is flushed as it lands, and a
    resumed judging run skips the ``(answer, judge_rep)`` pairs already scored.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

    @staticmethod
    def _key(obs_key: str, judge_rep: int) -> str:
        return f"{obs_key}::jr{judge_rep}"

    def load(self) -> dict[str, "Rating"]:
        """Return existing verdicts keyed by ``obs_key::jr{judge_rep}`` (last wins)."""
        from cafe.judging.ratings import Rating

        done: dict[str, Rating] = {}
        if not os.path.exists(self.path):
            return done
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = Rating(**json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    continue  # tolerate a torn final line
                done[self._key(r.obs_key, r.judge_rep)] = r
        return done

    def append(self, rating: "Rating") -> None:
        """Append one verdict, flushed to disk so a crash loses at most this one."""
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rating.to_dict(), default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
