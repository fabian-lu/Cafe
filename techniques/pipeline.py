"""The seeded system for the CAFE web app — the HotpotQA RAG evaluation pipeline.

This is the same compound system as `examples/evaluation/evaluation.ipynb`: a RAG pipeline over a
pooled HotpotQA knowledge base, with three retrieval levels (none / dense / dense_rerank), a model
knob on generation, and a placebo (negative-control) stage.

The pooled KB embeddings are **bundled** (`kb_emb.npy`, built with bge-m3) so the app boots without
re-embedding; only the query is embedded live at retrieval time. Edit this file (or mount your own
`techniques/` folder) to change the system under test — no web-app code changes needed.

Requires OPENROUTER_API_KEY (query embeddings) + OLLAMA_API_KEY (generation/rerank) in the env.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import litellm

import cafe

litellm.suppress_debug_info = True
_HERE = Path(__file__).resolve().parent

# ── config (mirrors the evaluation notebook) ──────────────────────────────────
SYSTEM_MODELS = {                       # levels of the `generate.model` factor
    "gemma3-4b":    "ollama_cloud/gemma3:4b",
    "gpt-oss-20b":  "ollama_cloud/gpt-oss:20b",
    "gpt-oss-120b": "ollama_cloud/gpt-oss:120b",
}
RERANK_MODEL = "ollama_cloud/gpt-oss:20b"      # LLM reranker for dense_rerank
EMBED_MODEL = "openrouter/baai/bge-m3"         # query embedder (KB is pre-embedded)
TOP_K = 2                                       # paragraphs fed to the generator
RETRIEVE_K = 12                                 # candidate pool the reranker reorders
EMBED_COST_USD = 1e-5                            # nominal per-query embedding cost

# ── the pooled knowledge base (rebuilt from the bundled questions, in the SAME order the bundled
#    embeddings were computed: question order, dedup by title) ──────────────────
_questions = json.load(open(_HERE / "hotpot_questions.json"))


def _paragraph(title, sentences):
    return f"{title}. " + " ".join(s.strip() for s in sentences)


_kb = {}
for _ex in _questions:
    for _t, _s in zip(_ex["context"]["title"], _ex["context"]["sentences"]):
        _kb.setdefault(_t, _paragraph(_t, _s))
KB_TITLES = list(_kb)
KB_DOCS = [_kb[t] for t in KB_TITLES]
KB_EMB = np.load(_HERE / "kb_emb.npy")
assert len(KB_EMB) == len(KB_DOCS), f"KB cache mismatch: {len(KB_EMB)} emb vs {len(KB_DOCS)} docs"


def _embed(texts):
    resp = litellm.embedding(model=EMBED_MODEL, input=texts)
    arr = np.asarray([d["embedding"] for d in resp["data"]], dtype=np.float32)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)


def retrieve(query, k):
    """Top-k (title, text, score) from the pooled KB by cosine similarity (query embedded live)."""
    q = _embed([query])[0]
    sims = KB_EMB @ q
    idx = np.argsort(-sims)[:k]
    return [(KB_TITLES[i], KB_DOCS[i], float(sims[i])) for i in idx]


async def _llm_rerank(query, candidates, k):
    listing = "\n".join(f"[{i}] {text}" for i, (_, text, _) in enumerate(candidates, 1))
    prompt = (f"Question: {query}\n\nPassages:\n{listing}\n\n"
              f"Rank the passages by how well they help answer the question. "
              f"Reply with ONLY the {k} best passage numbers, best first, e.g. '3 1'.")
    raw = await cafe.complete(RERANK_MODEL, [{"role": "user", "content": prompt}], temperature=0.0)
    picks, seen, order = [int(n) for n in re.findall(r"\d+", raw)], set(), []
    for p in picks:
        if 1 <= p <= len(candidates) and p not in seen:
            seen.add(p); order.append(candidates[p - 1])
    return (order or candidates)[:k]


pipe = cafe.Pipeline()


# ── retrieve: three techniques → the `retrieve` factor ──
@pipe.technique("retrieve", "none", description="No retrieval — the model answers from its own "
                "parametric memory only. The known-worst anchor.")
async def retrieve_none(ctx, query):
    return []                                  # parametric memory only (the anchor)


@pipe.technique("retrieve", "dense", cost_usd=EMBED_COST_USD,
                description="Dense retrieval: embed the query (bge-m3) and take the top-k KB paragraphs "
                "by cosine similarity.")
async def retrieve_dense(ctx, query):
    return [text for _, text, _ in retrieve(query, TOP_K)]


@pipe.technique("retrieve", "dense_rerank", cost_usd=EMBED_COST_USD,
                description="Dense retrieval + an LLM reranker (RankGPT-style) that reorders the top-k "
                "candidates before feeding the generator.")
async def retrieve_dense_rerank(ctx, query):
    cands = retrieve(query, RETRIEVE_K)
    order = await _llm_rerank(query, cands, TOP_K)
    return [text for _, text, _ in order]


# ── generate: one technique; `model` is the knob → the `generate.model` factor ──
@pipe.technique("generate", "answer",
                description="Answer the question grounded in the retrieved context. The `model` knob "
                "selects which LLM (gemma3-4b / gpt-oss-20b / gpt-oss-120b).")
async def generate_answer(ctx, query, docs, model="gpt-oss-20b"):
    context = "\n\n".join(docs) if docs else "(no documents retrieved)"
    return await cafe.complete(SYSTEM_MODELS.get(model, model), [
        {"role": "system", "content": "Answer the question using the provided context when it helps. "
         "If the context is missing or insufficient, still give your best answer from your own "
         "knowledge — always attempt an answer even if unsure, and never refuse. Be concise and factual."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"},
    ], temperature=0.3)


# ── finalize: the PLACEBO — both levels return the answer UNCHANGED (negative control) ──
@pipe.technique("finalize", "on", description="Placebo (negative control): returns the answer "
                "unchanged. True effect is zero — CAFE must call it non-significant.")
async def finalize_on(ctx, answer):
    return answer


@pipe.technique("finalize", "off", description="Placebo (negative control): returns the answer "
                "unchanged. The other level of the no-op control.")
async def finalize_off(ctx, answer):
    return answer


# ── compose: retrieve → generate → finalize ──
@pipe.compose
async def run(config, item, ctx):
    docs = await ctx.run("retrieve", query=item["text"])
    answer = await ctx.run("generate", query=item["text"], docs=docs)
    return await ctx.run("finalize", answer=answer)


# The HotpotQA questions as dataset items (text + gold reference) — the web app can seed these into a
# Dataset so a study runs on the real eval questions. Exposed for the seeder script.
DATASET_ITEMS = [
    {"id": str(q.get("id", i)), "text": q["question"], "reference": q["answer"]}
    for i, q in enumerate(_questions)
]
