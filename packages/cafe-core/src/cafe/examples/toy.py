"""A neutral, dependency-free example compound system.

This stands in for a real compound AI system so the engine can be exercised with
no API keys. It simulates a tiny pipeline whose answer quality depends on two
factors — which model and which prompting strategy — plus realistic run-to-run
noise (the thing CAFE exists to measure). It is intentionally domain-neutral:
no DIVA, no specific provider.

The "quality" it reports here is only a stand-in so later slices (judging, stats)
have a signal to attribute. In a real study, quality comes from the judge, not
the system.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

# Ground-truth effect sizes baked into the simulation, so a recovered analysis
# can be checked against them. Larger model helps a lot; chain-of-thought helps
# a little; together a small positive interaction.
_MODEL_EFFECT = {"small": 0.0, "large": 1.4}
_PROMPT_EFFECT = {"plain": 0.0, "cot": 0.6}
_INTERACTION = {("large", "cot"): 0.3}
_BASELINE = 2.5  # on a notional 1..5 quality scale
_NOISE_SD = 0.6  # run-to-run nondeterminism


async def toy_system(config: dict[str, Any], item: Any) -> dict[str, Any]:
    """Pretend to answer ``item`` under ``config``; return output + sim metadata."""
    model = str(config.get("model", "small"))
    prompt = str(config.get("prompt", "plain"))
    temperature = float(config.get("temperature", 0.0))

    # Simulate latency: bigger model is slower.
    latency = 0.01 + (0.02 if model == "large" else 0.005)
    await asyncio.sleep(latency)

    # Simulated, unseeded noise => genuine variance across replications.
    mean = (
        _BASELINE
        + _MODEL_EFFECT.get(model, 0.0)
        + _PROMPT_EFFECT.get(prompt, 0.0)
        + _INTERACTION.get((model, prompt), 0.0)
    )
    score = random.gauss(mean, _NOISE_SD * (1.0 + temperature))
    score = max(1.0, min(5.0, score))  # clamp to the notional scale

    question = item["text"] if isinstance(item, dict) and "text" in item else str(item)
    cost = (0.004 if model == "large" else 0.001) * (1.5 if prompt == "cot" else 1.0)

    return {
        "output": f"[{model}/{prompt}] answer to: {question[:60]}",
        "sim_quality": round(score, 3),
        "cost_usd": round(cost, 4),
        "latency_s": round(latency, 4),
    }


def build_example_study():
    """A small 2-factor study over the toy system, ready to run."""
    from cafe.study import Factor, FactorType, Study

    inputs = [
        {"id": f"q{i}", "text": t}
        for i, t in enumerate(
            [
                "What is the capital of France?",
                "Summarize the theory of relativity in one sentence.",
                "Why is the sky blue?",
                "Give a recipe for plain pancakes.",
            ]
        )
    ]
    return Study(
        name="toy-2factor",
        system=toy_system,
        factors=[
            Factor("model", ["small", "large"], FactorType.categorical),
            Factor("prompt", ["plain", "cot"], FactorType.categorical),
        ],
        dataset=inputs,
        design="full_factorial",
        replications=3,
    )
