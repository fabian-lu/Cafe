"""Study, Factor — the user-facing description of an experiment.

A Study is the black box (the system under test) plus the factors to vary and
the inputs to run it on. It carries no execution or persistence logic itself;
:func:`cafe.execution.run_study` consumes it.

This deliberately replaces DIVA's fixed ``Stage``/``PIPELINE_ORDER`` model: CAFE
does not model pipeline topology. A factor is just a named axis with levels; the
system reads the chosen levels from the ``config`` dict and does whatever it wants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FactorType(str, Enum):
    """How a factor is treated by design generation and (later) statistics."""

    categorical = "categorical"  # unordered: reranker in {none, cross_encoder}
    ordinal = "ordinal"          # ordered: effort in {low, med, high}
    continuous = "continuous"    # numeric knob: temperature, top_k


@dataclass
class Factor:
    """One axis of the experiment.

    ``levels`` is the set of values this factor can take. Full factorial visits
    every combination of every factor's levels — there is no limit on the number
    of factors or levels (that limit only applies to fractional/screening designs).
    """

    name: str
    levels: list[Any]
    type: FactorType = FactorType.categorical

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("factor name must be a non-empty string")
        self.levels = list(self.levels)
        if len(self.levels) == 0:
            raise ValueError(f"factor {self.name!r} must have at least one level")
        if isinstance(self.type, str):
            self.type = FactorType(self.type)


@dataclass
class Study:
    """A black box + the factors to vary over it + the inputs to evaluate.

    ``system`` is anything runnable as ``run(config, item) -> output`` — a plain
    callable (sync or async) or an object with such a method. See
    :mod:`cafe.system`.

    ``inputs`` is the evaluation set: one element per item. An item may be any
    value; if it is a mapping with an ``"id"`` key, that id is used for
    idempotency/resume, otherwise the item's position is used.
    """

    name: str
    system: Any
    factors: list[Factor] = field(default_factory=list)
    inputs: list[Any] = field(default_factory=list)
    design: str = "full_factorial"
    replications: int = 1

    def __post_init__(self) -> None:
        names = [f.name for f in self.factors]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate factor names: {names}")
        if self.replications < 1:
            raise ValueError("replications must be >= 1")

    # Convenience: run this study synchronously. For async contexts (notebooks
    # already inside an event loop), use ``await cafe.run_study(study)`` instead.
    def run(self, **kwargs: Any):
        import asyncio

        from cafe.execution import run_study

        return asyncio.run(run_study(self, **kwargs))
