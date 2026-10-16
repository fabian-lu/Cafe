from cafe import Factor, FactorType, Study, full_factorial, generate, size
from cafe.design import Config  # noqa: F401


def _study(design="full_factorial", **kw):
    return Study(
        name="t",
        system=lambda config, item: "x",
        factors=[
            Factor("model", ["small", "large"]),
            Factor("prompt", ["plain", "cot", "react"]),
        ],
        dataset=["q"],
        design=design,
        **kw,
    )


def test_full_factorial_is_cartesian_product():
    configs = full_factorial(_study().factors)
    assert len(configs) == 2 * 3
    # every combination is present and unique
    seen = {(c["model"], c["prompt"]) for c in configs}
    assert len(seen) == 6


def test_no_factors_yields_one_empty_config():
    assert full_factorial([]) == [{}]


def test_size_matches_generate():
    s = _study()
    assert size(s) == len(generate(s))


def test_single_requires_pinned_factors():
    pinned = Study(
        name="t",
        system=lambda config, item: "x",
        factors=[Factor("model", ["large"]), Factor("prompt", ["cot"])],
        dataset=["q"],
        design="single",
    )
    assert generate(pinned) == [{"model": "large", "prompt": "cot"}]


def test_single_rejects_varying_factor():
    import pytest

    with pytest.raises(ValueError):
        generate(_study(design="single"))


def test_factor_validation():
    import pytest

    with pytest.raises(ValueError):
        Factor("empty", [])
    f = Factor("ord", ["a", "b"], "ordinal")
    assert f.type is FactorType.ordinal


def test_reserved_factor_names_rejected():
    """A factor named like a bookkeeping column would crash the stats ('rep', 'input_id',
    'verdict'), silently merge configurations ('error', 'reasoning', 'judge_rep') or be
    overwritten in Evaluation.records() ('output', 'tokens', 'energy_wh', 'question')."""
    import pytest

    reserved = (
        "input_id rep judge_rep verdict reasoning error question reference output "
        "elapsed_s cost_usd tokens energy_wh judge_system judge_prompt judge_raw"
    )
    for name in reserved.split():
        with pytest.raises(ValueError, match="reserved"):
            Factor(name, ["a", "b"])
    for ok in ("reps", "error_rate", "model.output", "Rep"):  # near-misses stay allowed
        Factor(ok, ["a", "b"])


def test_duplicate_factor_names_rejected():
    import pytest

    with pytest.raises(ValueError):
        Study(
            name="t",
            system=lambda c, i: "x",
            factors=[Factor("m", ["a"]), Factor("m", ["b"])],
            dataset=["q"],
        )
