# CAFE — evaluation protocol (EMNLP demo)

**Status: draft for red-teaming (2026-07-03). Nothing run yet.** This is the *design*; approve it
before we build the notebooks or spend any human-rating effort. The notebooks that execute it live
next to this file (`examples/evaluation/`).

---

## 0. What we are (and aren't) evaluating

Two different claims, and only the second is an "evaluation":

- **Evaluating a system *with* CAFE** — "we ran CAFE on a RAG system, it says config C is best."
  This is a **showcase**. It shows the tool runs; it can't be checked, because *nobody knows the true
  best config* for a real system.
- **Evaluating CAFE *itself*** — "when CAFE says *factor X drives quality* / *this gap is real* /
  *the judge is trustworthy*, that claim is **correct**." This needs **ground truth**, and it's the
  actual evaluation.

The only hard EMNLP rule (verified at <https://2026.emnlp.org/calls/demos/>): *"submissions that do
not report any form of evaluation may be desk rejected."* No human study is mandated; "any form"
counts. Everything below is **well above** that bar — by design, because the evaluation doubles as
the paper's novelty evidence.

## 1. Design principle — one real pipeline, two kinds of factor

We do **not** build a separate toy for the evaluation. We run **one real study on a proper
pipeline**, and choose the factors so that some carry **known ground truth** (the evaluation anchors)
and some are **genuinely open** (the discovery / showcase):

| factor | role | levels | ground truth | used for |
|---|---|---|---|---|
| `retrieval` | **anchor — must help** | `off` · `realistic` · `oracle` | known **ordering** `oracle ≥ realistic ≥ off` (multi-hop answers aren't in parametric memory) | recovery (E1) |
| `instruction_phrasing` | **anchor — must ≈ null** | `A` · `B` (two meaning-equivalent paraphrases) | ≈ **null** effect | recovery (E1) |
| `model` | **open — the finding** | `small` · `large` (gpt-oss:20b / 120b) | unknown — CAFE tells us | showcase + holdout (E4) |

Base design **3 × 2 × 2 = 12 configs**; add `top_k {2,4}` or `retriever {embed_A, embed_B}` as a 4th
open factor only if the token budget allows. The anchors make this an *evaluation of the tool*: we
know what CAFE **should** conclude (retrieval huge + correctly ordered; phrasing ≈ null), so we can
check it does. `model` is the genuine experiment a user would run.

## 2. Pipeline & dataset — DECIDED

- **Dataset:** **HotpotQA** (`hotpotqa/hotpot_qa`, `distractor` config). Multi-hop, and every row ships
  its own paragraphs + a gold-sentence key — giving us three levels of ground truth in one dataset:
  the **corpus** (`context`), **relevance labels** (`supporting_facts`), and the **short answer**
  (→ objective EM/F1). ~**40–60 questions** (enough for stable mixed-effects; see `study.check()`),
  `replications=2` so the noise/significance story bites.
- **Knowledge base — pooled, realistic.** We do **not** retrieve only from a question's own 10
  paragraphs (that retrieval is trivially easy — nothing for the retriever/top_k to affect). Instead
  we **pool every question's `context` paragraphs into one shared KB** (~40–60 × 10 ≈ 400–600 docs,
  dedup by title). Now each question's 2 gold paragraphs sit among hundreds of distractors, so
  retrieval quality genuinely matters — a realistic RAG setting (HotpotQA "fullwiki-lite").
- **Retriever — embeddings (realistic).** Embed the pooled KB once; retrieve top-k by cosine
  similarity to the question. (Embedding model = small impl decision: a local `sentence-transformers`
  model e.g. `all-MiniLM-L6-v2` / `bge-small` — free, deterministic, standard RAG baseline — or an
  Ollama embedding model to stay in-ecosystem. Default: `sentence-transformers`.)
- **Pipeline (composed / Mode B):** `retrieve → generate`.
  - `retrieval=off` → generate with **no** context (parametric memory only).
  - `retrieval=realistic` → embedding top-k from the **pooled** KB (may miss a gold paragraph).
  - `retrieval=oracle` → feed exactly the gold paragraphs (from `supporting_facts`) — the upper bound.
- **No web-search factor here.** It's realistic but would **break the retrieval anchor** (if the model
  can web-search, `retrieval=off` no longer must-fail). Keep this study RAG-clean; a web-search
  showcase can be a separate, non-ground-truth notebook.
- **Dual scoring:** every answer is scored **by the LLM judge (CAFE, the thing under test)** *and* by
  **EM/F1 vs the gold `answer`** (objective ground truth). E1 checks CAFE's judge-based attribution
  reaches the same conclusion as the objective EM/F1-based attribution — recovery **without** needing
  humans.

## 3. The evaluation moves

Each move states the **CAFE claim it tests**, the **procedure**, the **pass criterion**, and **where
it lands in the paper**. E1–E2 use ground truth (the core); E3 validates the ratings; E4 covers
generalization; E5 is already done.

### E1 — Recovery (construct validity) — *the headline*
- **Claim tested:** CAFE's variance attribution / significance correctly identifies *which factor
  matters and which doesn't*.
- **Procedure:** run the full study; read `result.effects` / `result.clmm`. Check the **anchors**:
  the *must-help* factor is **significant, positive, large** (top partial η² / a big Cohen's d /
  a clearly-signed CLMM coefficient); the *must-not-matter* factor is **non-significant / near-zero**.
- **Pass criterion:** correct **significance decision** and **sign** on both anchors; the *must-help*
  factor ranks at/near the top of the variance attribution. (We claim sign + rough magnitude, not an
  exact effect size — see Limitations.)
- **Paper:** §Evaluation, the headline sentence + the effects table/forest plot.

### E2 — Null / false-positive calibration
- **Claim tested:** CAFE does **not** manufacture significance where there is none (specificity).
- **Procedure (analytic null, airtight):** take the real ratings and **permute the config labels**
  across answers (so any factor→quality link is destroyed by construction), refit, and record whether
  any factor comes out "significant." Repeat the permutation **K = 500–1000** times; the fraction of
  runs with ≥1 significant factor at α = 0.05 is the **empirical false-positive rate**.
- **Pass criterion:** empirical FPR ≈ the nominal α (≈ 5%), not wildly above. This is a real
  calibration curve, cheap (no new LLM calls — it re-analyses existing verdicts).
- **Paper:** one sentence + a small number ("under label permutation, CAFE's false-positive rate was
  X% at α=0.05"). Very high credibility-per-word.

### E3 — Judge ↔ human agreement (validates the input)
- **Claim tested:** the LLM judge's ratings — which everything above rests on — track human judgment.
- **Procedure:**
  1. Sample **~60–100 answers** from the study, **stratified to span the quality range** (α is
     meaningless if every answer is a 5 — reuse the spread logic from `03_human_and_irr`).
  2. **≥2 (ideally 3) human raters** score them on the *same rubric* the judge used. Use
     `cafe.answer_sheet(result, "sheet.csv", raters=[...])` → team fills it in → `cafe.human_ratings`.
  3. Report **human↔human α first** (the *ceiling* — do humans even agree?), then **judge↔human α**
     (does the judge reach that ceiling?), with **bootstrap 95% CIs**. `cafe.reliability(result,
     human=…)` computes both.
- **Pass criterion:** human↔human α is respectable (≳ 0.6–0.7, else the task is too subjective to
  fault the judge); judge↔human α is **within CI of / not far below** the human ceiling.
- **Paper:** §Evaluation, the IRR result — "the judge agrees with human experts at α = X [CI], vs a
  human–human ceiling of Y." This is the reliability evidence for the ratings.

### E4 — Holdout / within-distribution stability
- **Claim tested:** CAFE's "best config" is **not overfit to the specific N questions** used.
- **Scope (important — your point):** this is **within-distribution** only. Test questions are a fresh
  **draw from the same task/benchmark**, not "similar content." We claim *"best-on-sample-A also wins
  on sample-B from the same distribution,"* **not** cross-task transfer (which would be false — the
  best config depends on the questions asked).
- **Procedure:** split the questions 50/50 (or k-fold). Pick the best config (and the factor ranking)
  on split A; check it still wins / the ranking holds on split B.
- **Pass criterion:** the winning config (or top factor) agrees across splits **when the gap was
  significant**. If a split disagrees, cross-check that CAFE's own significance test had already
  flagged that gap as **not significant** — i.e. CAFE told you not to trust it. (Holdout failure on a
  *non-significant* gap is a *success* for CAFE's honesty, not a failure.)
- **Paper:** a sentence; secondary to E1–E3. State the within-distribution scope explicitly.

### E5 — Implementation correctness (already done — just report it)
- Unit tests + **reference cross-checks**: Krippendorff α pinned bit-exact to the `krippendorff`
  library (incl. the Hayes & Krippendorff 2007 case), logistic odds ratios hand-verified, CLMM via R
  `ordinal`, the fractional-design aliasing capped correctly. **Paper:** one sentence + appendix
  pointer. This backs the "the math is right" claim that E1's real-pipeline recovery only shows to
  sign+magnitude.

## 4. Human-rating logistics (E3) — the only effort cost

- **Who:** you + 1–2 teammates (domain-aware raters are fine and standard for judge↔human α).
- **How much:** ~60–100 items × 2–3 raters ≈ **a couple of hours per person**.
- **How:** `answer_sheet()` writes a CSV (question, answer, reference, blank score); raters fill the
  `score` column in Excel/Sheets **independently** (don't discuss — that inflates agreement);
  `human_ratings()` reads them back. Tooling is already shipped and tested.
- **Guard:** rate on the **same rubric** the judge used; include a few obviously-bad answers so the
  scale is exercised.

## 5. What lands in the 6-page paper (§Evaluation, ~1–1.25 p)

1. One-line study description (pipeline, factors incl. which are anchors, dataset, N, reps).
2. **E1 recovery** — the effects table/forest + "CAFE recovered the retrieval effect (η²=…, p=…) and
   correctly found the cosmetic factor null." *(headline)*
3. **E2 null** — "false-positive rate X% at α=0.05 under label permutation."
4. **E3 IRR** — "judge↔human α = … [CI] vs human ceiling … ."
5. **E4 holdout** — one sentence, within-distribution scope stated.
6. **E5** — one sentence on implementation validation + appendix.
7. **Comparison table** (separate, reuse `papers.csv`: RAGAS/ARES/LangSmith/Inspect — DoE? variance
   attribution? ordinal stats? human+judge? IRR?) — answers "how does it compare."

## 6. Limitations to state (turn weaknesses into credibility)

- Real-pipeline recovery validates **sign + rough magnitude**, not exact effect size (that's what E5
  and the appendix controlled check are for).
- "Best config" is **distribution-specific** — CAFE finds the best config for the questions you
  evaluate on; it makes **no** cross-task-transfer claim. Re-run it for a new task.
- Judge↔human α uses **domain-aware team raters**, not a crowd — a validity check on *this* judge for
  *this* task, not a claim about judges in general.

## 7. (Optional) airtight controlled recovery — appendix only, if time
A fully-synthetic mini-study with **exact** known effects (a factor that adds a known constant to
answer quality) to show CAFE recovers the *magnitude*, not just the sign. Nice-to-have; E1+E5 already
cover the claim for a demo.

## 8. Decisions
- ✅ **Dataset:** HotpotQA (distractor config), **pooled context → one shared KB** (§2).
- ✅ **Retriever:** embeddings (realistic RAG), not keyword/BM25 (§2).
- ✅ **Web-search:** excluded from the ground-truth study (would break the retrieval anchor); optional
  separate showcase later.
- ✅ **Null anchor:** `instruction_phrasing {A,B}` (always-defined paraphrase pair), with the
  permutation test (E2) as the rigorous null.
- ⏳ **Embedding model:** `sentence-transformers` (`all-MiniLM-L6-v2` / `bge-small`) vs an Ollama
  embedding model — default `sentence-transformers`; confirm.
- ⏳ **N questions & budget:** 40–60 questions × 12 configs × 2 reps ≈ 1,000–1,400 generations + judge
  calls. OK? (Bump/trim to taste; `study.plan()`/`preflight()` will estimate before running.)
- ⏳ **Raters (E3):** who, and 2 or 3?
- ⏳ **Scope:** confirm 1 combined study (showcase + E1–E4), E5 already done, controlled recovery (§7)
  optional appendix.

## 9. Build plan (once §8 ⏳ items are confirmed)
1. `examples/evaluation/01_build_kb.ipynb` — load HotpotQA, pool contexts, embed the KB, sanity-check
   that gold paragraphs are retrievable (retriever recall@k on `supporting_facts`).
2. `examples/evaluation/02_study.ipynb` — the composed RAG pipeline + the 12-config study + dual
   scoring (CAFE judge + EM/F1); the showcase report/plots.
3. `examples/evaluation/03_evaluation.ipynb` — E1 recovery (judge vs EM/F1 attribution), E2 permutation
   null, E4 holdout; write the numbers for the paper's §Evaluation.
4. E3 (human α): `answer_sheet()` → team rates → `human_ratings()` → `reliability()`; folded into 03.
