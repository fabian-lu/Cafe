"""Statistics over judge ratings: attribute quality to factors, test significance.

Three layers: descriptive (means) → inferential (Gaussian mixed model, F/p, η²,
Cohen's d) → ordinal CLMM (R). The CLMM is the only piece needing an external
runtime (R); everything else is pure Python.
"""

from cafe.stats.descriptive import Attribution, attribute
from cafe.stats.inferential import Effects, fit_effects
from cafe.stats.ordinal import CLMMResult, check_r, fit_clmm
from cafe.stats.pareto import ParetoResult, pareto
from cafe.stats.reliability import (
    HumanRatings,
    Reliability,
    answer_sheet,
    human_ratings,
    krippendorff_alpha,
    reliability,
)

__all__ = [
    "Attribution",
    "attribute",
    "Effects",
    "fit_effects",
    "CLMMResult",
    "fit_clmm",
    "check_r",
    "pareto",
    "ParetoResult",
    "reliability",
    "Reliability",
    "human_ratings",
    "HumanRatings",
    "answer_sheet",
    "krippendorff_alpha",
]
