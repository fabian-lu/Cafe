"""A second, minimal system (no retrieval) — to demonstrate multi-pipeline discovery + selection.

A study can pick THIS pipeline instead of the RAG one. It varies the prompting style (concise vs
chain-of-thought) and the model. Discovered because it exposes a `pipe` with a `@pipe.compose`.
"""

from __future__ import annotations

import cafe

pipe = cafe.Pipeline()

MODELS = {
    "gpt-oss-20b":  "ollama_cloud/gpt-oss:20b",
    "gpt-oss-120b": "ollama_cloud/gpt-oss:120b",
    "gemma3-4b":    "ollama_cloud/gemma3:4b",
}


@pipe.technique("answer", "concise", description="Answer directly in one concise sentence.")
async def concise(ctx, query, model="gpt-oss-20b"):
    return await cafe.complete(MODELS.get(model, model), [
        {"role": "system", "content": "Answer in one concise, factual sentence."},
        {"role": "user", "content": query},
    ], temperature=0.3)


@pipe.technique("answer", "cot", description="Chain-of-thought: reason step by step, then answer.")
async def cot(ctx, query, model="gpt-oss-20b"):
    return await cafe.complete(MODELS.get(model, model), [
        {"role": "system", "content": "Reason step by step, then give the final answer on the last line."},
        {"role": "user", "content": query},
    ], temperature=0.3)


@pipe.compose
async def run(config, item, ctx):
    return await ctx.run("answer", query=item["text"])
