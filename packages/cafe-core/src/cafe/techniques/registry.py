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
