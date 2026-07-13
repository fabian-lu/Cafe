# Judging

The **judge** turns each answer into a score on your **rubric**. That score is what
CAFE attributes back to your factors, so the judge is load-bearing — this page covers
how to configure it, swap it, and check it.

## The LLM judge

`cafe.LLMJudge` scores answers with any LiteLLM-supported model using a
research-grounded prompt preset:

```python
judge = cafe.LLMJudge(
    model="ollama_cloud/gpt-oss:20b",   # any LiteLLM model id
    preset="reference_qa",              # the prompt style (below)
    temperature=0.0,                    # deterministic by default
)
```

Pass it to the study, alongside a rubric:

```python
study = cafe.Study(..., rubric=cafe.rubrics.CORRECTNESS_0_3, judge=judge)
result = study.evaluate()
```

### Presets

The preset chooses the prompt scaffold:

| Preset | Use it when | Reference? |
|---|---|---|
| `reference_qa` *(default)* | you have a gold answer to grade against | required |
| `single_answer` | open-ended quality, no gold answer | not used |
| `criterion` | grade against a stated criterion (faithfulness, safety, …) | not used |

`reference_qa` reads a `reference` from each dataset item; give your inputs a
`"reference"` key for it (`{"text": ..., "reference": ...}`). The others are
reference-free.

### Auditing every verdict

Each judge call is stored in full — the exact prompt sent and the raw model response
sit behind every score, so you can audit or debug any grade. CAFE also asks for a
structured (JSON) verdict where the model supports it and falls back to a `GRADE: <n>`
line otherwise (`structured="auto"`, the default; force with `True`/`False`).

## Rubrics

A rubric fixes the **scale** answers are scored on — and the scale decides which
statistical model runs (see [Interpreting results](../interpreting-results.md)). Six
are built in:

| Rubric | Scale | Type |
|---|---|---|
| `CORRECTNESS_0_3` | 0–3 correctness | ordinal |
| `ANSWER_QUALITY_1_5` | 1–5 quality | ordinal |
| `FAITHFULNESS_1_5` | 1–5 grounding | ordinal |
| `RELEVANCE_1_5` | 1–5 relevance | ordinal |
| `HELPFULNESS_0_10` | 0–10 helpfulness | numeric |
| `CORRECT_PASS_FAIL` | pass / fail | binary |

Define your own by listing its levels and scale type:

```python
tone = cafe.Rubric(
    name="tone",
    scale_type=cafe.ScaleType.ordinal,
    levels=[
        cafe.Level(0, "off",   "Wrong register or rude."),
        cafe.Level(1, "ok",    "Acceptable but flat."),
        cafe.Level(2, "great", "Warm, on-brand, appropriate."),
    ],
    instruction="Judge the TONE of the answer, ignoring factual correctness.",
)
```

## Judge self-consistency

An LLM judge can score the *same* answer differently on repeat calls. Set
`judge_replications` to have it grade each answer several times, then inspect the
spread:

```python
study = cafe.Study(..., judge=judge, judge_replications=3)
result = study.evaluate()
print(result.judge_stability().show())   # per-answer std dev + a summary
```

Repeated passes are averaged before the factor models (so they do not masquerade as
extra data), but the raw spread is kept as a diagnostic. Note that repetitions only
tell you something at a non-zero judge temperature — a deterministic judge will not
vary.

## Generate once, judge many ways

Generating answers is the expensive part. `rejudge()` re-scores the **same answers**
with a different judge, rubric, or repetition count — nothing is regenerated:

```python
result = study.evaluate()

cheaper = result.rejudge(cafe.LLMJudge(model="ollama_cloud/gpt-oss:20b"))
binary  = result.rejudge(judge, rubric=cafe.rubrics.CORRECT_PASS_FAIL)
noisy   = result.rejudge(judge, repetitions=3)
```

This is also the basis for **judge ↔ judge** agreement:

```python
cafe.reliability(raters={"120b": result, "20b": result.rejudge(judge_20b)})
```

## A custom (non-LLM) judge

A judge is anything with a `model` attribute and an async `score()` returning a
`JudgeOutput`. This lets you plug in a deterministic scorer, an EM/F1 metric, or an
external service — no API key required:

```python
class KeywordJudge:
    model = "keyword-match"
    async def score(self, rubric, question, answer, reference=None):
        v = 3 if reference and reference.lower() in str(answer).lower() else 0
        return cafe.JudgeOutput(value=v, value_numeric=v, reasoning="",
                                prompt="", raw_response=str(v))

study = cafe.Study(..., rubric=cafe.rubrics.CORRECTNESS_0_3, judge=KeywordJudge())
```

The full walkthrough — presets, custom rubrics, rejudging, and stability — is in the
[Judging modes notebook](../notebooks/04_judging_modes.ipynb).
