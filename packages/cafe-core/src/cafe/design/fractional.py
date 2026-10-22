"""Fractional factorial designs — estimate the same main effects for far fewer runs.

A full factorial of ``k`` two-level factors costs ``2^k`` configurations, which
explodes (7 factors = 128). A **regular fractional factorial** runs a chosen subset
``2^(k-p)`` that still estimates the main effects, by deliberately *confounding*
(aliasing) them with high-order interactions you're willing to assume are negligible.

The price is **resolution**: the length of the shortest "word" in the design's
defining relation. Resolution III aliases main effects with 2-factor interactions
(good for *screening* which factors matter); IV with 3-factor; V keeps main effects
and 2-factor interactions clear. CAFE picks **minimum-aberration** generators (fewest
short defining words, which maximizes resolution) and **reports the resolution and
alias structure** so the trade-off is explicit and honest.

Only **two-level** factors are supported here (that's what regular fractions are
defined for); mixed-level factors must use the full factorial.
"""

from __future__ import annotations

import functools
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


#: Cap on the generator search's work (defining words evaluated). Deterministic — it
#: counts operations, not seconds — and keeps the worst case to tens of milliseconds.
_SEARCH_BUDGET = 1_000_000


def _word_counts(gens: list[int], b: int) -> list[int]:
    """``counts[L]`` = number of defining words of length ``L`` for generator columns
    ``gens`` (bitmasks over the ``b`` basic factors; added factor ``t`` is bit ``b + t``)."""
    counts = [0] * (b + len(gens) + 1)
    words: list[int] = []
    for t, g in enumerate(gens):
        w = g | (1 << (b + t))
        new = [w] + [w ^ x for x in words]
        for x in new:
            counts[x.bit_count()] += 1
        words += new
    return counts


@functools.lru_cache(maxsize=None)
def _choose_generators(b: int, p: int) -> tuple[tuple[int, ...], ...]:
    """Pick ``p`` distinct generators (subsets of the ``b`` basic factors, size ≥ 2)
    with **minimum aberration**: the fewest shortest defining words, compared length by
    length. That maximizes the resolution and breaks ties between designs of equal
    resolution the textbook way.

    Branch and bound over generator sets, starting from odd-size generators (every
    defining word then has even length ≥ 4: the classical resolution-IV construction,
    available whenever ``k ≤ runs/2``). Word counts only grow as generators are added,
    so a partial set already no better than the best complete one is pruned. Relabeling
    the basic factors doesn't change the word lengths, so the heaviest generator can be
    fixed to ``{0, ..., s-1}``. The search is exhaustive for the common shapes; past
    :data:`_SEARCH_BUDGET` it stops and keeps the best design found.
    """
    cols = [m for m in range(1, 1 << b) if m.bit_count() >= 2]
    order = sorted(cols, key=lambda m: (m.bit_count() % 2 == 0, -m.bit_count(), m))
    best_gens = order[:p]
    best = _word_counts(best_gens, b)
    work = 0

    class _Budget(Exception):
        pass

    def add(chosen: list[int], words: list[int], counts: list[int], gen: int,
            cands: list[int], nxt: int) -> None:
        """Add ``gen`` to ``chosen``; record the set if complete, else try every next
        generator from ``cands[nxt:]``."""
        nonlocal best, best_gens, work
        work += len(words) + 1
        if work > _SEARCH_BUDGET:
            raise _Budget
        w = gen | (1 << (b + len(chosen)))
        new = [w] + [w ^ x for x in words]
        counts = counts[:]
        for x in new:
            counts[x.bit_count()] += 1
        if counts >= best:  # adding generators only adds words: can't beat the best
            return
        chosen = chosen + [gen]
        if len(chosen) == p:
            best, best_gens = counts, chosen
            return
        words = words + new
        for i in range(nxt, len(cands) - (p - len(chosen) - 1)):
            add(chosen, words, counts, cands[i], cands, i + 1)

    try:
        for s in range(b, 1, -1):
            heaviest = (1 << s) - 1
            rest = [m for m in order if m.bit_count() <= s and m != heaviest]
            add([], [], [0] * (b + p + 1), heaviest, rest, 0)
    except _Budget:
        pass

    gens = [tuple(i for i in range(b) if m >> i & 1) for m in best_gens]
    return tuple(sorted(gens, key=lambda s: (-len(s), s)))


#: Most factors a fractional design may add to its basic ones: the defining relation has
#: 2^p - 1 words, so beyond this the design can't be analysed in reasonable memory.
_MAX_ADDED_FACTORS = 18


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

    The first ``log2(runs)`` factors are the basic factors (a full factorial); every
    other factor is "added" as the product of a generator subset of them.

    Parameters
    ----------
    factors:
        The two-level factors (at least 3).
    runs:
        The fraction's size, a power of two below ``2^k``. Default: the smallest
        design that can place every factor (a resolution-III screening design when
        nearly saturated).
    generators:
        ``{added_factor: [basic_factor, ...]}`` to specify the aliasing yourself. If
        ``None``, CAFE searches for the **minimum-aberration** generators (the fewest
        shortest defining words, so maximum resolution first; ties go to the design
        with fewer words of the next length). The search is deterministic and gives the
        textbook maximum resolution for every design up to 128 runs. For larger
        designs it stops after a fixed amount of work and keeps the best design found,
        which is at least resolution IV whenever ``runs >= 2k`` but may fall short of
        the maximum when only a few factors are added.

    Returns
    -------
    FractionalDesign
        The configs plus the generators, defining words, resolution and main-effect
        aliases (``.show()`` prints them).

    Raises
    ------
    ValueError
        If a factor doesn't have exactly two levels, there are fewer than 3 factors,
        ``runs`` isn't a power of two ≥ 4 below ``2^k``, the factors don't fit in
        ``runs``, or ``generators`` misses an added factor.
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
    if len(added) > _MAX_ADDED_FACTORS:
        # the defining relation has 2^p - 1 words (p = added factors): 2^26 for 31 factors in
        # 32 runs — enumerating them would exhaust memory long before producing a design
        raise ValueError(
            f"{k} factors in {runs} runs adds {len(added)} factors to {b} basic ones; its "
            f"defining relation (2^{len(added)} words) is too large to analyse — use at most "
            f"{b + _MAX_ADDED_FACTORS} factors here, or more runs"
        )

    # Assign each added factor a generator (a subset of ≥ 2 basic factors), chosen by a
    # minimum-aberration search. (Taking the largest subsets first is not enough: XOR of
    # ABCDE and ABCF is DEF, so greedy gave resolution III where IV/V designs exist.)
    if generators is None:
        if len(added) > (1 << b) - b - 1:
            raise ValueError(f"cannot place {k} factors in {runs} runs; increase runs")
        chosen = _choose_generators(b, len(added))
        gen_idx = {added[t]: chosen[t] for t in range(len(added))}
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
