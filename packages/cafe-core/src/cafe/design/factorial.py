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


def _fractional(factors: list[Factor], **options: Any) -> list[Config]:
    from cafe.design.fractional import fractional_factorial

    return fractional_factorial(factors, **options)


_GENERATORS = {
    "full_factorial": full_factorial,
    "single": single,
    "fractional": _fractional,
}

#: The ``design_options`` each design takes (passed to it as keyword arguments).
_OPTIONS: dict[str, tuple[str, ...]] = {
    "full_factorial": (),
    "single": (),
    "fractional": ("runs", "generators"),
}


def generate(study: Study) -> list[Config]:
    """Expand a study's design into its list of configurations.

    Parameters
    ----------
    study:
        The study. Its ``design_options`` dict is passed to designs that take parameters:
        ``fractional`` accepts ``runs`` / ``generators``; ``full_factorial`` and
        ``single`` take none.

    Returns
    -------
    list of dict
        One ``{factor: level}`` dict per configuration.

    Raises
    ------
    ValueError
        If the design is unknown, if ``design_options`` holds an option the design
        doesn't take (e.g. ``runs`` left over after switching a study from
        ``fractional`` back to ``full_factorial``: ignoring it could run many times the
        configurations it asked for), or if the design itself rejects the factors.
    """
    try:
        gen = _GENERATORS[study.design]
    except KeyError:
        raise ValueError(
            f"unknown design {study.design!r}; available: {sorted(_GENERATORS)}"
        ) from None
    options = getattr(study, "design_options", None) or {}
    takes = _OPTIONS.get(study.design, ())
    unknown = [k for k in options if k not in takes]
    if unknown:
        owner = next((d for d, opts in _OPTIONS.items() if set(unknown) <= set(opts)), None)
        raise ValueError(
            f"design={study.design!r} doesn't take design_options {unknown}"
            + (f", which are for design={owner!r}: remove them, or set design={owner!r}"
               if owner else f"; it takes {', '.join(takes) or 'none'}")
        )
    if options:
        return gen(study.factors, **options)
    return gen(study.factors)


def size(study: Study) -> int:
    """Number of configurations the design will produce (without running it).

    Raises
    ------
    ValueError
        Whenever :func:`generate` does, e.g. for ``design_options`` the design doesn't
        take.
    """
    return len(generate(study))
