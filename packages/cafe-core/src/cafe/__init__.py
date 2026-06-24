"""CAFE core — a design-of-experiments evaluation engine for compound AI systems.

CAFE *measures* compound AI systems; it does not implement them. You provide a
black box ``run(config, item) -> output`` and declare the factors to vary; CAFE
generates the design, executes every configuration with replication, and (in
later slices) judges and attributes the results.

Quick start (no API keys needed)::

    from cafe.examples import build_example_study
    results = build_example_study().run()
    print(results.summary())

Or define your own::

    import cafe

    async def my_system(config, item):
        return f"answer for {item} using {config['model']}"

    study = cafe.Study(
        name="my-study",
        system=my_system,
        factors=[cafe.Factor("model", ["a", "b"])],
        inputs=["question 1", "question 2"],
    )
    results = study.run()
"""

from cafe.design import full_factorial, generate, single, size
from cafe.execution import estimate, run_study
from cafe.results import Observation, Results, config_id, config_label
from cafe.study import Factor, FactorType, Study
from cafe.system import System, as_system

__version__ = "0.0.1"

__all__ = [
    "Study",
    "Factor",
    "FactorType",
    "System",
    "as_system",
    "run_study",
    "estimate",
    "Results",
    "Observation",
    "config_id",
    "config_label",
    "generate",
    "size",
    "full_factorial",
    "single",
    "__version__",
]
