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

from cafe import datasets
from cafe.design import full_factorial, generate, single, size
from cafe.evaluation import Evaluation, Preflight, evaluate, preflight
from cafe.execution import estimate, run_study
from cafe.judging import (
    JUDGE_PRESETS,
    Judge,
    JudgeOutput,
    LLMJudge,
    Rating,
    Ratings,
    build_judge_prompt,
    judge_results,
)
from cafe.llm import LLMError, complete
from cafe.execution.results import Observation, Results, config_id, config_label
from cafe.judging.rubric import ANSWER_QUALITY_1_5, Level, Rubric, ScaleType
from cafe.stats import (
    Attribution,
    CLMMResult,
    Effects,
    attribute,
    check_r,
    fit_clmm,
    fit_effects,
)
from cafe.study import Factor, FactorType, Study
from cafe.system import System, as_system

__version__ = "0.0.1"

__all__ = [
    # ── data ──
    "datasets",
    # ── define an experiment ──
    "Study",
    "Factor",
    "FactorType",
    "System",
    "as_system",
    # ── run it ──
    "evaluate",       # the complete pipeline: answers -> judge -> attribution
    "preflight",      # cheap pre-run check + cost estimate
    "run_study",      # lower-level: answers only
    "Evaluation",
    "Preflight",
    "Results",
    "Observation",
    "estimate",
    # ── designs ──
    "generate",
    "size",
    "full_factorial",
    "single",
    "config_id",
    "config_label",
    # ── judging ──
    "complete",
    "LLMError",
    "Rubric",
    "Level",
    "ScaleType",
    "ANSWER_QUALITY_1_5",
    "LLMJudge",
    "Judge",
    "JudgeOutput",
    "JUDGE_PRESETS",
    "judge_results",
    "build_judge_prompt",
    "Rating",
    "Ratings",
    # ── statistics ──
    "attribute",
    "Attribution",
    "fit_effects",
    "Effects",
    "fit_clmm",
    "CLMMResult",
    "check_r",
    "__version__",
]
