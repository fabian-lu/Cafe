"""The spec for one registered technique.

A *technique* is a unit of your compound system (a retriever, a reranker, a model
call, a verifier). Techniques are registered onto a :class:`~cafe.techniques.pipe.Pipeline`
you own — there is no global registry. A technique's keyword arguments *with defaults*
(``top_k``) are its tunable parameters and become parameter factors; arguments *without*
defaults (``query``) are runtime inputs you pass via ``ctx.run``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class TechniqueSpec:
    stage: str
    name: str
    fn: Callable[..., Any]
    params: dict[str, Any]      # tunable parameter -> default
    description: str = ""
    cost_usd: float = 0.0       # fixed cost charged each time this technique runs
    # Energy is strictly opt-in: None means "not declared" — no energy is computed or
    # reported for runs of this technique (0.0 would mean "declared to be free").
    energy_wh: float | None = None                # fixed Wh charged per run
    energy_wh_per_1k_tokens: float | None = None  # Wh per 1k tokens used inside the run
