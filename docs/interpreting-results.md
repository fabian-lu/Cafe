# Interpreting results

Defining and running a study is only half of CAFE. The payoff is the **report** —
which factor drives quality, by how much, and whether the difference is real. This
page walks through that output section by section.

## `run()` vs `evaluate()`

```python
results = study.run()        # answers only — the raw outputs + timing/cost
result  = study.evaluate()   # the full pipeline: answers → judge → attribution
```

- **`run()`** executes every configuration and returns a [`Results`][cafe.Results]
  object (outputs, latency, cost, errors). Use it when you only want the answers, or
  you will judge them later.
- **`evaluate()`** does the same, then scores each answer with your **rubric + judge**
  and fits the statistics. It returns an [`Evaluation`][cafe.Evaluation]. This is the
  one most users want — it needs a `rubric` and a `judge` on the study.

```python
study = cafe.Study(
    name="rag", system=pipe, factors=[...], dataset=[...],
    rubric=cafe.rubrics.CORRECTNESS_0_3,
    judge=cafe.LLMJudge(model="ollama_cloud/gpt-oss:20b"),
    replications=2,
)
result = study.evaluate()
print(result.report())        # the full statistical picture, below
```

!!! tip "Displaying vs. reporting"
    Just showing a result (`result` in a notebook, or `repr(result)`) prints only the
    fast **descriptive** layer, so it is instant. `result.report()` additionally *fits
    the models*, so the first call may take a moment (the ordinal model runs in R).

## The report, section by section

`report()` returns one string with a header and up to three analysis layers. Here is a
real report (abridged) from the bundled HotpotQA study — three factors, `retrieve`
(none / dense / dense\_rerank), `generate.model` (three sizes), and a placebo
`finalize` (on / off) — graded on a 0–2 correctness rubric.

### Header

```text
900 answers | 18 configs | 50 inputs | 900 ratings
best: finalize=on | generate.model=gpt-oss-120b | retrieve=dense_rerank
pipeline: 900 answers -> 900 judged (1 unparseable) -> 899 usable verdicts
```

The counts, the winning configuration, and how many verdicts survived parsing. A few
unparseable judge replies are normal; they are dropped, not guessed.

### 1 · Descriptive — means & best configuration

```text
overall mean quality: 1.31  (n=899)

per-factor marginal means:
  retrieve:        none 0.75 (n=300)   dense 1.48 (n=300)   dense_rerank 1.71 (n=299)
  generate.model:  gemma3-4b 1.08     gpt-oss-20b 1.35      gpt-oss-120b 1.51
  finalize:        off 1.30 (n=449)   on 1.32 (n=450)

best configuration: finalize=on | generate.model=gpt-oss-120b | retrieve=dense_rerank  (mean 1.78)
```

The immediate picture, no model assumed. **Marginal means** average a factor's level
over everything else — so `retrieve` climbs 0.75 → 1.48 → 1.71, while `finalize`
barely moves (1.30 vs 1.32). This is the *description*; the next layer decides which of
those gaps is real.

### 2 · Inferential — which factor matters, and is it significant

This is the core. For a numeric or ordinal rubric CAFE fits a linear mixed-effects
model with a per-input random effect, then reports a Type-II ANOVA:

```text
per-term effects  (F-test, p, partial eta^2;  'x' = interaction):
  term                             F          p    partial eta^2
  retrieve                    130.37     0.0000        0.228   ***
  generate.model               24.06     0.0000        0.052   ***
  retrieve x generate.model     4.17     0.0024        0.019   **
  generate.model x finalize     0.12     0.8858        0.000
  finalize                      0.17     0.6841        0.000
  retrieve x finalize           0.04     0.9612        0.000
  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

Read it row by row:

- **`p`** — the probability of seeing an effect this large if the factor did *nothing*.
  Small `p` (`< 0.05`) means the effect is unlikely to be noise. Here `retrieve` and
  `generate.model` are significant; `finalize` (the placebo) is correctly **not**
  (`p = 0.68`) — CAFE does not invent an effect that is not there.
- **`partial eta^2`** — the share of variance the factor explains, holding the others
  fixed. This is the "how much does it matter?" number: `retrieve` (0.228) dominates,
  `generate.model` (0.052) is a smaller real effect. Rough bands: 0.01 small, 0.06
  medium, 0.14 large.
- **`F`** and the **significance stars** are two more views of the same test.
- **`retrieve x generate.model`** is a significant **interaction** (`p = 0.002`): the
  two factors are not independent — retrieval helps the weaker model more than the
  strong one. A one-factor-at-a-time sweep cannot see this; the factorial design is
  exactly what surfaces it.

Below the table, **Cohen's d** gives each pairwise gap a standardized magnitude with a
confidence interval:

```text
effect sizes -- Cohen's d (magnitude of the gap; 0.2 small, 0.5 medium, 0.8 large):
  retrieve: dense_rerank vs none               d = +1.24   95% CI [+1.07, +1.42]
  retrieve: dense vs dense_rerank              d = -0.31   95% CI [-0.47, -0.15]
  finalize: off vs on                          d = -0.02   95% CI [-0.16, +0.11]
```

A CI that excludes 0 (like `dense_rerank vs none`) is a reliable gap; one that
straddles 0 (like `finalize`) is not.

!!! warning "Health notes"
    The report appends notes when a fit is fragile, e.g. *"model was near-singular
    (little between-question variance) — treat p-values as unstable; the effect sizes
    (Cohen's d) are the more reliable signal here."* Take these seriously: they tell
    you which numbers to trust when the design is small or sparse.

### 3 · Ordinal — the scale-correct model

When the rubric is **ordinal** (ordered categories like 0 < 1 < 2), CAFE also fits a
cumulative-link mixed model (CLMM) — the statistically correct model for ordered
verdicts, which the linear view only approximates:

```text
fixed effects (ordinal log-odds of a higher score; + = better):
  term                                                  estimate       p
  retrieve=dense_rerank                                   +1.149    0.0075   **
  retrieve=none                                           -3.944    0.0000   ***
  generate.model=gpt-oss-120b                             +1.250    0.0035   **
  retrieve=none x generate.model=gpt-oss-120b             +1.854    0.0006   ***
  finalize=on                                             -0.070    0.8524
```

Each estimate is the **log-odds** of landing in a higher score category, relative to
the baseline level. Positive = better. It confirms the story from the linear layer on
the correct scale: retrieval is correctly ordered (`none` strongly negative,
`dense_rerank` positive), the base model helps, and `finalize` stays null.

## Which model runs for which rubric

The rubric's scale type selects the model automatically — you do not choose:

| Rubric scale | Model fitted | Report section |
|---|---|---|
| `ordinal` (e.g. `CORRECTNESS_0_3`) | linear approximation **+** cumulative-link mixed model | Inferential + Ordinal |
| `numeric` (e.g. `HELPFULNESS_0_10`) | linear (Gaussian) mixed model | Linear |
| `binary` (e.g. `CORRECT_PASS_FAIL`) | logistic mixed model (log-odds, odds ratios) | Logistic |

## Beyond the report

`report()` is the one-call summary. The same numbers are available programmatically,
plus a few outputs the report does not include:

```python
result.attribution      # descriptive means + best configuration
result.effects          # the F / p / partial η² / Cohen's d model
result.clmm             # the ordinal CLMM (ordinal rubrics)

# judge ↔ human (or judge ↔ judge) agreement — Krippendorff's α
cafe.reliability(raters={"judge": result, "human": human_result})

# cost / latency trade-off: configs that are best value, not just best quality
cafe.pareto(result)

# how consistently the judge scored the same answer across repetitions
result.rejudge(judge, repetitions=3).judge_stability()
```

See the [Judging modes](notebooks/04_judging_modes.ipynb),
[Human ratings & IRR](notebooks/03_human_and_irr.ipynb), and
[Cost vs quality](notebooks/05_cost_quality.ipynb) notebooks for these in full.
