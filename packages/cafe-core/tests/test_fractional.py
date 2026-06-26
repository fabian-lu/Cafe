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
