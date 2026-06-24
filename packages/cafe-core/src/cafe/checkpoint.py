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
from typing import Iterator

from cafe.results import Observation


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
