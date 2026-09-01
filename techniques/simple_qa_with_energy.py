"""simple_qa with per-model ENERGY declarations — demonstrates opt-in energy tracking.

Unlike `simple_qa.py` (kept as the energy-free counterpart), the model here is the
*technique* (not a parameter), because energy coefficients are declared per technique:
each model carries its own `energy_wh_per_1k_tokens`. The prompting style is a tunable
parameter (`answer.style`) instead.

Energy coefficients are EDUCATED GUESSES, not measurements — Wh per 1k OUTPUT-side
tokens, folding together active parameter count, H100-class batched serving
(~0.3-0.6 J per token for small dense models), and datacenter overhead (PUE ~1.2).
CAFE never guesses these numbers itself; whoever writes the pipeline declares them,
and they are meant to be challenged/refined right here.
"""

from __future__ import annotations

import cafe

pipe = cafe.Pipeline()

PROMPTS = {
    "concise": "Answer in one concise, factual sentence.",
    "cot": "Reason step by step, then give the final answer on the last line.",
}


async def _ask(model: str, style: str, query: str) -> str:
    return await cafe.complete(model, [
        {"role": "system", "content": PROMPTS[style]},
        {"role": "user", "content": query},
    ], temperature=0.3)


@pipe.technique(
    "answer", "gemma4-31b",
    description="Gemma 4 31B (dense).",
    energy_wh_per_1k_tokens=0.30,  # 31B dense: ~6-8x the compute of a 4B-class model per token
)
async def gemma(ctx, query, style="concise"):
    return await _ask("ollama_cloud/gemma4:31b", style, query)


@pipe.technique(
    "answer", "gpt-oss-20b",
    description="GPT-OSS 20B (MoE, ~3.6B active params).",
    energy_wh_per_1k_tokens=0.06,  # ~3.6B ACTIVE params ≈ dense-4B-class compute, slightly more memory traffic
)
async def gptoss20(ctx, query, style="concise"):
    return await _ask("ollama_cloud/gpt-oss:20b", style, query)


@pipe.technique(
    "answer", "gpt-oss-120b",
    description="GPT-OSS 120B (MoE, ~5.1B active params).",
    energy_wh_per_1k_tokens=0.15,  # ~5.1B active, but the 120B weight footprint spans more GPUs → more overhead
)
async def gptoss120(ctx, query, style="concise"):
    return await _ask("ollama_cloud/gpt-oss:120b", style, query)


@pipe.compose
async def run(config, item, ctx):
    return await ctx.run("answer", query=item["text"])
