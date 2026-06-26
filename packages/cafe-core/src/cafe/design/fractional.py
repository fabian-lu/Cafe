"""Fractional factorial designs — estimate the same main effects for far fewer runs.

A full factorial of ``k`` two-level factors costs ``2^k`` configurations, which
explodes (7 factors = 128). A **regular fractional factorial** runs a chosen subset
``2^(k-p)`` that still estimates the main effects, by deliberately *confounding*
(aliasing) them with high-order interactions you're willing to assume are negligible.

The price is **resolution**: the length of the shortest "word" in the design's
defining relation. Resolution III aliases main effects with 2-factor interactions
(good for *screening* which factors matter); IV with 3-factor; V keeps main effects
and 2-factor interactions clear. CAFE picks generators that maximize resolution and
**reports the resolution and alias structure** so the trade-off is explicit and honest.

Only **two-level** factors are supported here (that's what regular fractions are
defined for); mixed-level factors must use the full factorial.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

from cafe.study import Factor

Config = dict[str, Any]
_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}


@dataclass
class FractionalDesign:
    """A generated fractional factorial: the configs plus its defining properties."""

    configs: list[Config]
    factor_names: list[str]
    runs: int
    full_runs: int
    resolution: int | None
    generators: dict[str, list[str]] = field(default_factory=dict)   # added factor -> basic factors
    defining_words: list[list[str]] = field(default_factory=list)
    aliases: dict[str, list[str]] = field(default_factory=dict)      # main effect -> confounded terms

    @property
    def resolution_roman(self) -> str:
        return _ROMAN.get(self.resolution, str(self.resolution)) if self.resolution else "—"

    def show(self) -> str:
        saving = 100 * (1 - self.runs / self.full_runs)
        lines = [
            f"Fractional factorial 2^({len(self.factor_names)}-{int(math.log2(self.full_runs // self.runs))})"
            f"  —  resolution {self.resolution_roman}",
            "",
            f"  runs: {self.runs} of {self.full_runs} full  ({saving:.0f}% fewer)",
        ]
        if self.generators:
            lines.append("")
            lines.append("  generators (added factor = product of basic factors):")
            for f, basics in self.generators.items():
                lines.append(f"     {f} = {''.join(basics)}")
        if self.aliases:
            lines.append("")
            lines.append("  main-effect aliases (assumed negligible):")
            for f, al in self.aliases.items():
                if al:
                    lines.append(f"     {f} = {' = '.join(al)}")
        res = self.resolution
        if res is not None:
            note = {
                3: "main effects aliased with 2-factor interactions (screening)",
                4: "main effects clear of 2-factor interactions; 2FIs aliased with each other",
            }.get(res, "main effects and 2-factor interactions estimable")
            lines.append("")
            lines.append(f"  resolution {self.resolution_roman}: {note}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:  # pragma: no cover
        return f"<pre>{self.show()}</pre>"

    def __repr__(self) -> str:
        return (
            f"FractionalDesign(2^({len(self.factor_names)}-"
            f"{int(math.log2(self.full_runs // self.runs))}), "
            f"res {self.resolution_roman}, {self.runs}/{self.full_runs} runs)"
        )


def _full_relation(words: list[frozenset[int]]) -> set[frozenset[int]]:
    """Every non-empty XOR-combination of the generator words (the defining relation)."""
    full: set[frozenset[int]] = set()
    p = len(words)
    for r in range(1, p + 1):
        for combo in itertools.combinations(range(p), r):
            w: frozenset[int] = frozenset()
            for c in combo:
                w = w ^ words[c]
            if w:
                full.add(w)
    return full


def fractional_factorial_design(
    factors: list[Factor],
    *,
    runs: int | None = None,
    generators: dict[str, list[str]] | None = None,
) -> FractionalDesign:
    """Build a resolution-maximizing 2^(k-p) fractional factorial for 2-level factors.

    ``runs`` (a power of two) sets the fraction; default is the smallest design that
    can place every factor (a saturated, resolution-III screening design). Pass
    ``generators`` (``{added_factor: [basic_factor, ...]}``) to specify the aliasing
    yourself instead of letting CAFE choose.
    """
    bad = [f.name for f in factors if len(f.levels) != 2]
    if bad:
        raise ValueError(
            "fractional factorial supports only two-level factors; these are not: "
            f"{bad}. Use the full factorial, or give each factor exactly two levels."
        )
    k = len(factors)
    if k < 3:
        raise ValueError("fractional factorial needs at least 3 factors; use full_factorial")
    full_runs = 2 ** k

    if runs is None:
        b = next(bb for bb in range(2, k + 1) if (2 ** bb) - 1 >= k)
        runs = 2 ** b
    else:
        if runs & (runs - 1) or runs < 4:
            raise ValueError(f"runs must be a power of two ≥ 4; got {runs}")
        b = int(round(math.log2(runs)))
    if runs >= full_runs:
        raise ValueError(
            f"runs={runs} is not a fraction of the full {full_runs}; use full_factorial"
        )

    names = [f.name for f in factors]
    basic, added = names[:b], names[b:]

    # Assign each added factor a generator (a subset of basic factors), largest first
    # to push the defining words as long as possible (higher resolution).
    if generators is None:
        pool = [
            combo for size in range(b, 1, -1)
            for combo in itertools.combinations(range(b), size)
        ]
        if len(added) > len(pool):
            raise ValueError(f"cannot place {k} factors in {runs} runs; increase runs")
        gen_idx = {added[t]: pool[t] for t in range(len(added))}
    else:
        name_to_basic = {n: i for i, n in enumerate(basic)}
        gen_idx = {}
        for f in added:
            if f not in generators:
                raise ValueError(f"no generator given for added factor {f!r}")
            gen_idx[f] = tuple(name_to_basic[g] for g in generators[f])

    fidx = {n: i for i, n in enumerate(names)}

    # Build the design matrix: full factorial over basic factors in ±1, added
    # columns are the products of their generator's basic columns.
    configs: list[Config] = []
    for row in itertools.product((-1, 1), repeat=b):
        signs = {basic[i]: row[i] for i in range(b)}
        for f, gen in gen_idx.items():
            s = 1
            for i in gen:
                s *= row[i]
            signs[f] = s
        cfg = {f.name: f.levels[0 if signs[f.name] < 0 else 1] for f in factors}
        configs.append(cfg)

    # Defining relation + resolution + main-effect aliases.
    words = [frozenset((fidx[f],)) | frozenset(gen_idx[f]) for f in added]
    relation = _full_relation(words)
    resolution = min((len(w) for w in relation), default=None)
    defining_words = [sorted(names[i] for i in w) for w in sorted(relation, key=len)]

    aliases: dict[str, list[str]] = {}
    for f in names:
        fset = frozenset((fidx[f],))
        al = []
        for w in relation:
            term = fset ^ w
            if 1 <= len(term) <= 2:  # report aliasing with main effects / 2FIs
                al.append("".join(sorted(names[i] for i in term)))
        aliases[f] = sorted(set(al))

    return FractionalDesign(
        configs=configs,
        factor_names=names,
        runs=runs,
        full_runs=full_runs,
        resolution=resolution,
        generators={f: [basic[i] for i in gen_idx[f]] for f in added},
        defining_words=defining_words,
        aliases=aliases,
    )


def fractional_factorial(
    factors: list[Factor],
    *,
    runs: int | None = None,
    generators: dict[str, list[str]] | None = None,
) -> list[Config]:
    """The configurations of a fractional factorial (see
    :func:`fractional_factorial_design` for the full design report)."""
    return fractional_factorial_design(factors, runs=runs, generators=generators).configs
