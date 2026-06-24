"""Design generation: turn factors into the list of configurations to run.

Slice 1 ships the two ends of the design ladder:

- ``single`` — one configuration (the system as-is; the on-ramp / regression case).
  Requires every factor to have exactly one level.
- ``full_factorial`` — the Cartesian product of all factors' levels. No limit on
  the number of factors or levels; cost is the product of the level counts.

Fractional factorial, screening, and optimal designs come in later slices.
"""

from __future__ import annotations

from itertools import product
from typing import Any

from cafe.study import Factor, Study

Config = dict[str, Any]


def full_factorial(factors: list[Factor]) -> list[Config]:
    """Every combination of every factor's levels."""
    if not factors:
        return [{}]
    names = [f.name for f in factors]
    level_lists = [f.levels for f in factors]
    return [dict(zip(names, combo)) for combo in product(*level_lists)]


def single(factors: list[Factor]) -> list[Config]:
    """A single configuration. Each factor must be pinned to one level."""
    multi = [f.name for f in factors if len(f.levels) != 1]
    if multi:
        raise ValueError(
            "single-config design requires exactly one level per factor; "
            f"these vary: {multi}"
        )
    return [{f.name: f.levels[0] for f in factors}]


_GENERATORS = {
    "full_factorial": full_factorial,
    "single": single,
}


def generate(study: Study) -> list[Config]:
    """Expand a study's design into its list of configurations."""
    try:
        gen = _GENERATORS[study.design]
    except KeyError:
        raise ValueError(
            f"unknown design {study.design!r}; available: {sorted(_GENERATORS)}"
        ) from None
    return gen(study.factors)


def size(study: Study) -> int:
    """Number of configurations the design will produce (without running it)."""
    return len(generate(study))
