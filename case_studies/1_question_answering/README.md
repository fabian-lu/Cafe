# Case study 1 · Question answering with retrieval (HotpotQA)

**The question:** in a retrieval-augmented (RAG) pipeline, what actually drives answer quality:
the retrieval, the model, or the combination? And is each difference real, or noise?

This is the study from the CAFE paper (EMNLP 2026 System Demonstrations), re-run with the current
package.

## Design

| | |
|---|---|
| Questions | 50 hard multi-hop questions from HotpotQA (`data/hotpotqa_questions.json`) |
| Factor `retrieval` | `none` · `dense` · `dense_with_reranker` |
| Factor `model` | `gemma-3-4b` · `gpt-oss-20b` · `gpt-oss-120b` |
| Factor `placebo` | `on` · `off`: both do nothing, a negative control CAFE must find non-significant |
| Size | 3 × 3 × 2 = 18 configurations × 50 questions = 900 answers |
| Rubric | `answer_correctness`: 0 incorrect · 1 partially correct · 2 correct (ordinal) |
| Judge | DeepSeek-V4-Pro, a different model family from all three systems |
| Filter | `question_type`: `bridge` · `comparison` |

Fixed settings: retrieval embeds the question with bge-m3 and compares it with 500 pre-embedded
Wikipedia paragraphs; the reranker is GPT-OSS 20B; the model sees 2 paragraphs.

## Files

| File | What it is |
|---|---|
| `pipeline.py` | The system under test: retrieval → model → placebo. The web app runs the same file. |
| `question_answering_study.ipynb` | The study: data, factors, rubric, judge, run, report, save. |
| `data/` | Questions, the paragraph collection and its embeddings (see `data/SOURCE.md`). |
| `results/question_answering_results.json` | The saved results; import them in the web app (Studies → Import study). |

## How to run

From the repository root, with API keys for Ollama Cloud and OpenRouter in `.env`:

```bash
pip install -e "packages/cafe-core[all]"
cd case_studies/1_question_answering
jupyter lab question_answering_study.ipynb
```

The full run takes roughly an hour. It saves its progress as it goes, so an interrupted run
continues where it stopped.

## Cost and energy figures

The models run on Ollama Cloud (a subscription) and OpenRouter. For the cost/quality analysis every
model is charged at its OpenRouter list price (October 2026), set at the top of `pipeline.py`.

The energy figures are **estimates**, declared per model in Wh per 1,000 tokens. They scale with the
compute a model spends per token, which depends mostly on its active parameters (GPT-OSS 20B and
120B are mixture-of-experts models with about 3.6 B and 5.1 B active parameters; Gemma 3 4B is a
dense 4 B model), plus serving overhead for larger models. They are meant to compare models with
each other, not as measurements.

## Results

| | |
|---|---|
| Retrieval: none / dense / dense with reranker | 0.78 / 1.55 / 1.73 (partial η² .35) |
| Model: Gemma 3 4B / GPT-OSS 20B / GPT-OSS 120B | 1.17 / 1.36 / 1.53 (partial η² .06) |
| Retrieval × model | significant (ordinal model, p = .01) |
| Placebo | no effect (p = .96, d = 0.01) |
| Best configurations | GPT-OSS 20B + reranker 1.80 ≈ GPT-OSS 120B + reranker 1.78 |

- **Retrieval drives quality far more than the model.** The reranker adds a small but reliable step
  over plain dense retrieval (d = 0.27, 95% CI [0.11, 0.43]).
- **Retrieval helps the small model most.** Without retrieval the gap between Gemma 3 4B and
  GPT-OSS 120B is 0.66; with the reranker it is 0.18. Good retrieval partly substitutes for a bigger
  model, which is why the 20B model with the reranker is as good as the 120B.
- **The placebo comes out as nothing**, as it must: CAFE doesn't invent effects.
- **Question difficulty is large** (ICC .37 on the scores, .62 on the latent scale of the ordinal model).
- The re-run **replicates the paper** (paper: retrieval 0.75 / 1.48 / 1.71, model 1.08 / 1.35 / 1.51).
