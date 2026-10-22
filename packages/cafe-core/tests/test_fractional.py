"""Fractional factorial designs."""

import pytest

import cafe
from cafe.design.fractional import fractional_factorial_design


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


def test_too_many_added_factors_is_a_clear_error_not_a_memory_blowup():
    # 31 two-level factors in 32 runs: the defining relation would have 2^26 words
    factors = [cafe.Factor(f"f{i}", ["lo", "hi"]) for i in range(31)]
    with pytest.raises(ValueError, match="too large to analyse"):
        cafe.fractional_factorial_design(factors, runs=32)
