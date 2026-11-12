"""Fractional factorial designs."""

import itertools
import math

import pytest

import cafe
from cafe.design.fractional import FractionalDesign, fractional_factorial_design


def _factors(letters):
    return [cafe.Factor(c, [0, 1]) for c in letters]


def test_saturated_7_factors_is_8_runs_res_iii():
    d = fractional_factorial_design(_factors("ABCDEFG"))
    assert d.runs == 8 and d.full_runs == 128
    assert d.resolution == 3
    # all rows distinct
    assert len({tuple(sorted(c.items())) for c in d.configs}) == 8


def test_half_fraction_5_factors_is_resolution_v():
    d = fractional_factorial_design(_factors("ABCDE"), runs=16)
    assert d.runs == 16
    assert d.resolution == 5  # E = ABCD is the classic resolution-V design


@pytest.mark.parametrize(
    "k, runs, want",
    [
        (6, 16, 4),  # 2^(6-2)_IV
        (7, 16, 4),  # 2^(7-3)_IV
        (8, None, 4),  # 2^(8-4)_IV, the default runs for 8 factors
        (7, 32, 4),  # 2^(7-2)_IV
        (9, 32, 4),  # 2^(9-4)_IV
        (16, 32, 4),  # 2^(16-11)_IV, the largest resolution-IV design in 32 runs
        (17, 32, 3),  # beyond runs/2 factors only resolution III exists
        (8, 64, 5),  # 2^(8-2)_V
        (9, 128, 6),  # 2^(9-2)_VI
        (10, 128, 5),  # 2^(10-3)_V
        (11, 128, 5),  # 2^(11-4)_V
    ],
)
def test_auto_generators_reach_textbook_resolution(k, runs, want):
    """Largest-subsets-first generators gave resolution III for every shape here (only
    right for 17 factors in 32 runs); CAFE must reach the maximum in the standard tables."""
    d = fractional_factorial_design(_factors("ABCDEFGHIJKLMNOPQ"[:k]), runs=runs)
    assert d.resolution == want
    assert len({tuple(sorted(c.items())) for c in d.configs}) == d.runs
    if want >= 4:  # no main effect aliased with a main effect or 2-factor interaction
        assert not any(d.aliases.values())


@pytest.mark.parametrize(
    "k, runs, lengths",
    [
        (6, 16, [4, 4, 4]),
        (7, 32, [4, 5, 5]),
        (9, 64, [4, 5, 5, 5, 5, 6, 6]),
    ],
)
def test_auto_generators_are_minimum_aberration(k, runs, lengths):
    """Among designs of equal resolution, the fewest shortest words (Chen, Sun & Wu 1993)."""
    d = fractional_factorial_design(_factors("ABCDEFGHI"[:k]), runs=runs)
    assert sorted(len(w) for w in d.defining_words) == lengths


def test_main_effects_have_aliases():
    d = fractional_factorial_design(_factors("ABCDEFG"))
    # in a resolution-III design every main effect is aliased with some 2FI
    assert any(d.aliases[f] for f in d.factor_names)


def test_two_level_only():
    with pytest.raises(ValueError, match="two-level"):
        fractional_factorial_design([cafe.Factor("A", [0, 1]), cafe.Factor("B", [1, 2, 3])])


def test_runs_must_be_a_real_fraction():
    with pytest.raises(ValueError, match="full_factorial"):
        fractional_factorial_design(_factors("ABC"), runs=8)  # 8 == full for 3 factors


def test_study_uses_fractional_design():
    study = cafe.Study(
        name="frac",
        system=lambda config, item: "x",
        factors=_factors("ABCDEFG"),
        dataset=["q"],
        design="fractional",
    )
    assert cafe.size(study) == 8
    assert len(cafe.generate(study)) == 8


def test_study_fractional_with_options():
    study = cafe.Study(
        name="frac",
        system=lambda config, item: "x",
        factors=_factors("ABCDE"),
        dataset=["q"],
        design="fractional",
        design_options={"runs": 16},
    )
    assert cafe.size(study) == 16


def _true_resolution(d):
    """The shortest word whose product column is constant over the design's runs (brute force)."""
    signs = [[1 if c[n] == 1 else -1 for n in d.factor_names] for c in d.configs]
    k = len(d.factor_names)
    for length in range(1, k + 1):
        for word in itertools.combinations(range(k), length):
            if len({math.prod(row[i] for i in word) for row in signs}) == 1:
                return length
    return None


def test_user_generators_give_the_reported_resolution():
    d = fractional_factorial_design(_factors("ABCDE"), runs=8,
                                    generators={"D": ["B", "A"], "E": ["A", "C"]})
    assert d.generators == {"D": ["B", "A"], "E": ["A", "C"]}
    assert d.resolution == _true_resolution(d) == 3
    assert len({tuple(sorted(c.items())) for c in d.configs}) == 8


@pytest.mark.parametrize(
    "generators, problem",
    [
        # A·B·A = B: D was the same column as B (true resolution II), reported as III
        ({"D": ["A", "B", "A"], "E": ["A", "C"]}, "repeats \\['A'\\]"),
        # D = A aliases two main effects; reported "main effects ... estimable"
        ({"D": ["A"], "E": ["B", "C"]}, "at least 2 basic factors"),
        # an empty product never varies D; reported resolution I "...estimable"
        ({"D": [], "E": ["B", "C"]}, "'D' would never vary"),
        # D = E: aliased main effects again, in either order
        ({"D": ["A", "B"], "E": ["B", "A"]}, "same product"),
        # bare KeyErrors before: an added factor, and a typo
        ({"D": ["A", "B"], "E": ["D", "C"]}, "only multiply basic factors, but names \\['D'\\]"),
        ({"D": ["A", "b"], "E": ["A", "C"]}, "only multiply basic factors, but names \\['b'\\]"),
        # silently ignored before: a generator for a basic factor
        ({"C": ["A", "B"], "D": ["A", "B"], "E": ["A", "C"]}, "only added factors take one"),
    ],
    ids=["repeated", "single", "empty", "duplicate", "added-factor", "typo", "basic-factor"],
)
def test_invalid_user_generators_are_refused(generators, problem):
    with pytest.raises(ValueError, match=problem):
        fractional_factorial_design(_factors("ABCDE"), runs=8, generators=generators)


def test_show_keeps_textbook_words_for_single_letter_names():
    shown = fractional_factorial_design(_factors("ABCDEFG")).show()
    assert "D = ABC" in shown
    assert "A = BE = CF = DG" in shown


def test_show_separates_multi_character_factor_names():
    names = ["model", "retrieve", "top_k", "sabotage", "rerank"]
    d = fractional_factorial_design([cafe.Factor(n, ["a", "b"]) for n in names], runs=8)
    shown = d.show()
    assert "sabotage = model×retrieve×top_k" in shown   # was "modelretrievetop_k"
    assert "rerank = model×retrieve = sabotage×top_k" in shown
    assert d.aliases["model"] == ["rerank×retrieve"]


def test_show_has_no_empty_alias_section_from_resolution_iv():
    for d in (fractional_factorial_design(_factors("ABCDEF"), runs=16),   # IV
              fractional_factorial_design(_factors("ABCDE"), runs=16)):   # V
        assert d.resolution >= 4
        assert "main-effect aliases" not in d.show()


@pytest.mark.parametrize("resolution", [1, 2])
def test_show_says_main_effects_are_not_estimable_below_resolution_iii(resolution):
    d = FractionalDesign(configs=[], factor_names=["A", "B", "C"], runs=4, full_runs=8,
                         resolution=resolution)
    note = d.show().splitlines()[-1]
    assert "not estimable" in note and "interactions estimable" not in note


def test_too_many_added_factors_is_a_clear_error_not_a_memory_blowup():
    # 31 two-level factors in 32 runs: the defining relation would have 2^26 words
    factors = [cafe.Factor(f"f{i}", ["lo", "hi"]) for i in range(31)]
    with pytest.raises(ValueError, match="too large to analyse"):
        cafe.fractional_factorial_design(factors, runs=32)
