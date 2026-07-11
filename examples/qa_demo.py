"""Screencast demo system — a minimal, no-retrieval QA pipeline with two stages.

During the demo: copy this into `techniques/`, hit **Reload** on the Techniques page, then build a
study on it. Two stages become two factors:

  * ``draft``  — how to produce the first answer   (3 levels: direct / stepwise / expert)
  * ``refine`` — whether to polish it              (2 levels: keep / polish; ``keep`` = control)

3 × 2 = 6 configurations — a real factorial (you can ask whether refining helps *and* whether it
helps more for some draft styles), but small enough to run in the demo. Uses a small Ollama model;
needs OLLAMA_API_KEY in .env.
"""

from __future__ import annotations

import cafe

pipe = cafe.Pipeline()

MODEL = "ollama_cloud/gemma3:4b"   # small + fast, keeps the demo run short


# ── Stage 1: draft — three prompting styles ──────────────────────────────────────────────
@pipe.technique("draft", "direct", description="Answer directly in one factual sentence.")
async def direct(ctx, query):
    return await cafe.complete(MODEL, [
        {"role": "system", "content": "Answer in one short, factual sentence."},
        {"role": "user", "content": query},
    ], temperature=0.2)


@pipe.technique("draft", "stepwise", description="Reason step by step, then answer on the last line.")
async def stepwise(ctx, query):
    return await cafe.complete(MODEL, [
        {"role": "system", "content": "Reason step by step, then give the final answer on the last line."},
        {"role": "user", "content": query},
    ], temperature=0.2)


@pipe.technique("draft", "expert", description="Answer as a meticulous domain expert.")
async def expert(ctx, query):
    return await cafe.complete(MODEL, [
        {"role": "system", "content": "You are a meticulous domain expert. Give the precise, correct answer."},
        {"role": "user", "content": query},
    ], temperature=0.2)


# ── Stage 2: refine — keep the draft (control) or polish it ───────────────────────────────
@pipe.technique("refine", "keep", description="Use the draft unchanged (control level).")
async def keep(ctx, query, draft):
    return draft


@pipe.technique("refine", "polish", description="Rewrite the draft into one crisp, correct sentence.")
async def polish(ctx, query, draft):
    return await cafe.complete(MODEL, [
        {"role": "system", "content": "Rewrite the draft answer into a single crisp, correct sentence."},
        {"role": "user", "content": f"Question: {query}\nDraft: {draft}"},
    ], temperature=0.2)


@pipe.compose
async def run(config, item, ctx):
    draft = await ctx.run("draft", query=item["text"])
    return await ctx.run("refine", query=item["text"], draft=draft)
