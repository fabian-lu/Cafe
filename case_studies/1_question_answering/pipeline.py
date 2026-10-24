"""The system under test: a question-answering pipeline with retrieval (RAG) over HotpotQA paragraphs.

    question → retrieval → model → placebo → answer

Each stage has several interchangeable techniques. CAFE turns every stage into a factor and
tests all combinations:

- retrieval: none · dense · dense_with_reranker
- model:     gemma-3-4b · gpt-oss-20b · gpt-oss-120b
- placebo:   on · off  (both do nothing; a negative control: CAFE must find no effect)

The web app runs this same file, so the notebook and the app evaluate the identical system.
Needs OLLAMA_API_KEY and OPENROUTER_API_KEY in the repo's .env.
"""

import json
import re
from pathlib import Path

import litellm
import numpy as np

import cafe

DATA = Path(__file__).parent / "data"
litellm.suppress_debug_info = True   # no "Provider List" line printed for every call


# ── Prices ───────────────────────────────────────────────────────────────────────────────────────
# Ollama Cloud doesn't report what a call costs, so we tell CAFE each model's market price
# (OpenRouter list price, October 2026). CAFE expects USD per 1,000 tokens, hence "per million / 1000".
# From then on, CAFE prices every call made through cafe.complete() with this model.
cafe.set_model_cost(model="openrouter/google/gemma-3-4b-it", per_1k_input=0.04 / 1000, per_1k_output=0.05 / 1000)
cafe.set_model_cost(model="ollama_cloud/gpt-oss:20b", per_1k_input=0.018 / 1000, per_1k_output=0.09 / 1000)
cafe.set_model_cost(model="ollama_cloud/gpt-oss:120b", per_1k_input=0.15 / 1000, per_1k_output=0.60 / 1000)


# ── The paragraph collection that retrieval searches ─────────────────────────────────────────────
PARAGRAPHS = json.load(open(DATA / "paragraphs.json"))              # 500 Wikipedia paragraphs
PARAGRAPH_EMBEDDINGS = np.load(DATA / "paragraph_embeddings.npy")   # one stored vector per paragraph
assert len(PARAGRAPHS) == len(PARAGRAPH_EMBEDDINGS), "paragraphs and embeddings are out of sync"

EMBEDDING_MODEL = "openrouter/baai/bge-m3"     # turns the question into a vector
EMBEDDING_COST_USD = 0.00001                   # per question; nominal, embedding one question is tiny
RERANKER_MODEL = "ollama_cloud/gpt-oss:20b"    # the LLM that picks the most helpful paragraphs
PASSAGES_PER_QUESTION = 2                      # how many paragraphs the model gets to see
CANDIDATES_FOR_RERANKER = 12                   # how many paragraphs the reranker chooses from

RERANK_PROMPT = """Question: {question}

Passages:
{numbered_passages}

Rank the passages by how well they help answer the question. Reply with ONLY the {keep} best passage
numbers, best first, e.g. '3 1'."""

ANSWER_INSTRUCTIONS = (
    "Answer the question using the provided context when it helps. If the context is missing or "
    "insufficient, still give your best answer from your own knowledge — always attempt an answer "
    "even if unsure, and never refuse. Be concise and factual."
)
QUESTION_WITH_CONTEXT = "Context:\n{context}\n\nQuestion: {question}\nAnswer:"


# A CAFE pipeline: a set of stages, each with interchangeable techniques.
pipe = cafe.Pipeline()

# @pipe.technique(stage=..., name=...) registers the function below it as one technique of a stage.
# All techniques of a stage become the levels of that stage's factor. Optional settings:
#   cost_usd                 a fixed cost per call, for things CAFE can't price itself
#   energy_wh_per_1k_tokens  estimated energy per 1,000 tokens (see the README); CAFE multiplies it
#                            by the tokens of every cafe.complete() call made inside the technique


# ── Stage 1: retrieval ───────────────────────────────────────────────────────────────────────────
@pipe.technique(stage="retrieval", name="none",
                description="No retrieval: the model answers from what it learned in training.")
async def no_retrieval(ctx, question):
    return []


# The embedding call goes to LiteLLM directly, not through cafe.complete(), so CAFE doesn't count
# its tokens or energy; its (tiny) cost is declared with cost_usd instead.
@pipe.technique(stage="retrieval", name="dense", cost_usd=EMBEDDING_COST_USD,
                description="Dense retrieval: the 2 paragraphs most similar to the question (bge-m3).")
async def dense_retrieval(ctx, question):
    # Embed the question and compare it with every stored paragraph embedding (cosine similarity).
    response = litellm.embedding(model=EMBEDDING_MODEL, input=[question])
    question_vector = np.asarray(response["data"][0]["embedding"], dtype=np.float32)
    similarity = PARAGRAPH_EMBEDDINGS @ (question_vector / np.linalg.norm(question_vector))

    most_similar = np.argsort(-similarity)[:PASSAGES_PER_QUESTION]
    return [PARAGRAPHS[i]["text"] for i in most_similar]


@pipe.technique(stage="retrieval", name="dense_with_reranker", cost_usd=EMBEDDING_COST_USD,
                energy_wh_per_1k_tokens=0.06,   # the reranker is GPT-OSS 20B
                description="Dense retrieval of 12 candidates, then an LLM picks the 2 most helpful.")
async def dense_retrieval_with_reranker(ctx, question):
    # 1. Dense retrieval as above, but keep 12 candidates.
    response = litellm.embedding(model=EMBEDDING_MODEL, input=[question])
    question_vector = np.asarray(response["data"][0]["embedding"], dtype=np.float32)
    similarity = PARAGRAPH_EMBEDDINGS @ (question_vector / np.linalg.norm(question_vector))
    candidates = [PARAGRAPHS[i]["text"] for i in np.argsort(-similarity)[:CANDIDATES_FOR_RERANKER]]

    # 2. An LLM reads the numbered candidates and names the most helpful ones (RankGPT-style).
    numbered_passages = "\n".join(f"[{number}] {text}" for number, text in enumerate(candidates, start=1))
    prompt = RERANK_PROMPT.format(question=question, numbered_passages=numbered_passages,
                                  keep=PASSAGES_PER_QUESTION)
    reply = await cafe.complete(model=RERANKER_MODEL, messages=[{"role": "user", "content": prompt}],
                                temperature=0.0, num_retries=3)   # retry if briefly rate-limited

    # 3. Keep the paragraphs it named, in its order; if the reply is unusable, keep the original order.
    chosen = []
    for number in (int(n) for n in re.findall(r"\d+", reply)):
        if 1 <= number <= len(candidates) and candidates[number - 1] not in chosen:
            chosen.append(candidates[number - 1])
    return (chosen or candidates)[:PASSAGES_PER_QUESTION]


# ── Stage 2: the model that writes the answer (one technique per model) ──────────────────────────
@pipe.technique(stage="model", name="gemma-3-4b", energy_wh_per_1k_tokens=0.05,
                description="Gemma 3 4B (Google, 4B parameters) writes the answer.")
async def gemma_3_4b(ctx, question, passages):
    context = "\n\n".join(passages) or "(no documents retrieved)"
    return await cafe.complete(
        model="openrouter/google/gemma-3-4b-it",   # retired on Ollama Cloud, so via OpenRouter
        messages=[{"role": "system", "content": ANSWER_INSTRUCTIONS},
                  {"role": "user", "content": QUESTION_WITH_CONTEXT.format(context=context, question=question)}],
        temperature=0.3,
        num_retries=3,   # retry if the provider is briefly rate-limited
    )


@pipe.technique(stage="model", name="gpt-oss-20b", energy_wh_per_1k_tokens=0.06,
                description="GPT-OSS 20B (OpenAI open-weight, mixture of experts) writes the answer.")
async def gpt_oss_20b(ctx, question, passages):
    context = "\n\n".join(passages) or "(no documents retrieved)"
    return await cafe.complete(
        model="ollama_cloud/gpt-oss:20b",
        messages=[{"role": "system", "content": ANSWER_INSTRUCTIONS},
                  {"role": "user", "content": QUESTION_WITH_CONTEXT.format(context=context, question=question)}],
        temperature=0.3,
        num_retries=3,
    )


@pipe.technique(stage="model", name="gpt-oss-120b", energy_wh_per_1k_tokens=0.15,
                description="GPT-OSS 120B (OpenAI open-weight, mixture of experts) writes the answer.")
async def gpt_oss_120b(ctx, question, passages):
    context = "\n\n".join(passages) or "(no documents retrieved)"
    return await cafe.complete(
        model="ollama_cloud/gpt-oss:120b",
        messages=[{"role": "system", "content": ANSWER_INSTRUCTIONS},
                  {"role": "user", "content": QUESTION_WITH_CONTEXT.format(context=context, question=question)}],
        temperature=0.3,
        num_retries=3,
    )


# ── Stage 3: the placebo (a negative control) ────────────────────────────────────────────────────
@pipe.technique(stage="placebo", name="on", description="Does nothing: returns the answer unchanged.")
async def placebo_on(ctx, answer):
    return answer


@pipe.technique(stage="placebo", name="off", description="Does nothing either. The true effect is "
                "zero, so CAFE must find this factor non-significant.")
async def placebo_off(ctx, answer):
    return answer


# ── How the stages connect ───────────────────────────────────────────────────────────────────────
# @pipe.compose marks the function that wires the stages together. CAFE calls it once for every
# configuration and question; ctx.run(stage=...) runs the technique the configuration chose for
# that stage (and records its time, tokens, cost and energy).
# CAFE always passes (config, item, ctx). config isn't needed here, because ctx.run reads it
# itself; you would use it for your own control flow, e.g. `if config["model"] == ...`.
@pipe.compose
async def question_answering(config, item, ctx):
    passages = await ctx.run(stage="retrieval", question=item["text"])
    answer = await ctx.run(stage="model", question=item["text"], passages=passages)
    return await ctx.run(stage="placebo", answer=answer)
