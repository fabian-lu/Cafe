"""The instrumented context handed to a composed system.

When your system calls ``await ctx.run("retriever", query=q)``, CAFE:
  - reads the active config to see which technique fills that stage,
  - resolves the technique's tunable params from the config,
  - runs it, **times it**, captures its **cost** and **output**,
  - appends a step to a **trace**, and **caches** the result so configurations that
    share an upstream sub-path don't recompute it.

The control flow around these calls is just your Python — branches, cascades,
loops — so the topology is entirely yours; CAFE only watches the calls.
"""

from __future__ import annotations

import json
import time
from typing import Any

from cafe.techniques import registry


def _cache_key(stage: str, name: str, params: dict, inputs: dict) -> str:
    payload = json.dumps(
        {"stage": stage, "name": name, "params": params, "inputs": inputs},
        sort_keys=True, default=str,
    )
    return payload


class Context:
    """Runs your techniques and records what happened. One per (config, item) run."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.trace: list[dict[str, Any]] = []
        self.total_cost: float = 0.0
        self._cache: dict[str, Any] = {}

    async def run(self, stage: str, **inputs: Any) -> Any:
        """Run whichever technique the config selected for ``stage`` and return its output."""
        name = self.config.get(stage)
        if name is None:
            raise KeyError(
                f"the config selects no technique for stage {stage!r} "
                f"(add a factor named {stage!r} to the study)"
            )
        spec = registry.get(stage, str(name))

        params = {
            pname: self.config.get(f"{stage}.{pname}", default)
            for pname, default in spec.params.items()
        }

        key = _cache_key(stage, str(name), params, inputs)
        if key in self._cache:
            hit = self._cache[key]
            self.trace.append({**hit["meta"], "cached": True})
            return hit["output"]

        t0 = time.monotonic()
        raw = await spec.fn(self, **inputs, **params)
        elapsed = round(time.monotonic() - t0, 5)

        # A technique may return a plain value, or a dict {"output":…, "cost_usd":…}.
        output, cost = raw, 0.0
        if isinstance(raw, dict) and "output" in raw:
            output = raw["output"]
            cost = float(raw.get("cost_usd", 0.0) or 0.0)

        meta = {
            "stage": stage,
            "technique": str(name),
            "params": params,
            "elapsed_s": elapsed,
            "cost_usd": cost,
            "cached": False,
        }
        self.total_cost += cost
        self.trace.append(meta)
        self._cache[key] = {"output": output, "meta": meta}
        return output
