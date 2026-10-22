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

This is the core. Every configuration answers the same questions, so for a numeric or
ordinal rubric CAFE reports a Type-II ANOVA **blocked by question**: question difficulty is
taken out of the error term and configurations are compared *within* each question (in a
balanced design these are exactly the tests of a random-intercept mixed model):

```text
verdict ~ question + (finalize + generate.model + retrieve)^2
  Type-II ANOVA, blocked by question, up to 2-way   (n=899, alpha=0.05)

per-term effects  (F-test, p, partial eta^2;  'x' = interaction):
  term                             F          p    partial eta^2
  retrieve                    207.02     0.0000        0.331   ***
  generate.model               38.25     0.0000        0.084   ***
  generate.model x retrieve     6.60     0.0000        0.031   ***
  finalize x generate.model     0.19     0.8240        0.000
  finalize                      0.26     0.6132        0.000
  finalize x retrieve           0.06     0.9381        0.000
  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

Read it row by row:

- **`p`** — the probability of seeing an effect this large if the factor did *nothing*.
  Small `p` (`< 0.05`) means the effect is unlikely to be noise. Here `retrieve` and
  `generate.model` are significant; `finalize` (the placebo) is correctly **not**
  (`p = 0.61`) — CAFE does not invent an effect that is not there.
- **`partial eta^2`** — the share of variance the factor explains, holding the others
  fixed (and with question difficulty taken out). This is the "how much does it matter?"
  number: `retrieve` (0.331) dominates, `generate.model` (0.084) is a smaller real effect.
  Rough bands: 0.01 small, 0.06 medium, 0.14 large.
- **`F`** and the **significance stars** are two more views of the same test.
- **`generate.model x retrieve`** is a significant **interaction** (`p < 0.001`): the
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
straddles 0 (like `finalize`) is not. (Cohen's d compares the raw scores of two levels, so
unlike the F-test it does not take question difficulty out.)

The **variance components** line shows how much the questions themselves differ, from
the random-intercept model `verdict ~ factors + (1 | question)`:

```text
variance components:  between-question sigma^2 = 0.207 · residual sigma^2 = 0.366 · ICC = 0.361
```

Here 36% of the variance that the factors leave unexplained is simply *which question* it
was — some questions are harder for every configuration. That is why blocking by question
matters: without it, this variation would count as noise and every F would be smaller.

!!! warning "Health notes"
    The report appends notes when something about a fit needs your attention — e.g.
    interactions that a fractional design can't separate, an ordinal or logistic model
    whose optimizer did not converge (its p-values are then withheld), or a
    between-question variance estimated at ≈ 0 (a boundary fit: the questions don't
    differ beyond what the factors explain). Read them: they tell you which numbers to
    trust when the design is small or sparse.

    A tiny study can also fit **exactly**: the factors and the questions explain every
    score, so there is zero residual variance and the F-tests are degenerate. The
    model line then ends in *"exact fit (zero residual variance)"*, a term with any effect
    shows `F = inf`, `p = 0`, partial eta^2 = 1, and a term with no effect shows `-` (not
    significant). Cohen's d is undefined too (zero spread), so compare the configuration
    means directly — and add questions if you expect real variation.

### 3 · Ordinal — the scale-correct model

When the rubric is **ordinal** (ordered categories like 0 < 1 < 2), CAFE also fits a
cumulative-link mixed model (CLMM) — the statistically correct model for ordered
verdicts, which the linear view only approximates. It first answers *which factor
matters* with one **likelihood-ratio test** per factor and interaction (Type II: the model
with every term that doesn't contain it, with vs. without the term):

```text
per-factor tests (likelihood ratio, Type II — does the factor matter at all?):
  term                             chi^2   df          p
  retrieve                       334.17    2     0.0000   ***
  generate.model                  80.57    2     0.0000   ***
  generate.model x retrieve       16.36    4     0.0026   **
  finalize                         0.50    1     0.4792
  finalize x generate.model        0.49    2     0.7835
  finalize x retrieve              0.05    2     0.9772
```

Same verdict as the linear layer — retrieval and the model matter, they interact, the
placebo `finalize` doesn't — now on the scale-correct model. (`result.clmm.factor_tests`;
skip them with `fit_clmm(..., lr_tests=False)` — they refit the model a few times. With
very few questions, roughly under 10, such tests are somewhat liberal.) Then, per level:

```text
fixed effects (ordinal log-odds of a higher score; + = better):
  term                                                  estimate       p
  retrieve=dense_rerank                                   +1.149    0.0075   **
  retrieve=none                                           -3.944    0.0000   ***
  generate.model=gpt-oss-120b                             +1.250    0.0035   **
  retrieve=none x generate.model=gpt-oss-120b             +1.854    0.0006   ***
  finalize=on                                             -0.070    0.8524
```

The `p` column rests on the CLMM's standard errors. If R's optimizer stops without
converging, those are not valid: they can come out about 100× too small, which turns a
factor with no effect into `p = 0.0000 ***`. CAFE then shows `-` for every p-value, adds
the note *"R's optimizer did not converge (nlminb: …)"*, and sets `result.clmm.converged`
to `False`. The check is the optimizer's own convergence code, which `ordinal::clmm`
reports but does not act on. For such a study, read the linear layer's tests and effect
sizes instead.

Each estimate is the **log-odds** of landing in a higher score category, relative to
the baseline level. Positive = better. It confirms the story from the linear layer on
the correct scale: retrieval is correctly ordered (`none` strongly negative,
`dense_rerank` positive), the base model helps, and `finalize` stays null.

Under the table the CLMM reports its per-question random intercept (R's `VarCorr()`):

```text
  variance components:  between-question σ_u² = 5.347 (σ_u = 2.312) · latent residual π²/3 = 3.290 · ICC = 0.619
```

`σ_u²` is how much the questions differ in difficulty on the model's latent log-odds
scale. The ICC, `σ_u² / (σ_u² + π²/3)`, is the share of the latent variance that lies
between questions (`π²/3` is the fixed residual variance of a logit link). The logistic
GLMM for binary rubrics prints the same line. When `σ_u` collapses to about 0, the report
adds a *near-singular* note: the questions don't differ beyond what the factors explain. That
is a boundary fit, not a failure — the estimates and p-values stay valid (they equal those of
the same model without a question effect). The numbers are also available as
`result.clmm.variance_components` / `result.logistic.variance_components`.

## Which model runs for which rubric

The rubric's scale type selects the model automatically — you do not choose:

| Rubric scale | Model fitted | Report section |
|---|---|---|
| `ordinal` (e.g. `CORRECTNESS_0_3`) | linear approximation **+** cumulative-link mixed model | Inferential + Ordinal |
| `numeric` (e.g. `HELPFULNESS_0_10`) | linear model, questions as blocks (+ random-intercept variance components) | Linear |
| `binary` (e.g. `CORRECT_PASS_FAIL`) | logistic mixed model (log-odds, odds ratios) | Logistic |

The logistic mixed model (R `lme4::glmer`) comes with lme4's own convergence checks. When
they fail (lme4: *"Model failed to converge with max|grad| = …"*), the model's standard
errors are not valid; they can come out about 100× too small, which turns a factor with no
effect into `p = 0.0000 ***`. The report then shows `-` for every p-value and adds the note
*"R's optimizer did not converge (lme4: …)"*, and `result.logistic.converged` is `False`.
Because lme4's checks also flag fits that are fine, CAFE first refits once with the
`bobyqa` optimizer and uses that fit if it passes (the report says so); only if it doesn't
are the p-values withheld. A boundary (singular) fit is not a convergence failure. For such
a study, read the descriptive pass rates instead.

Like the CLMM, the logistic GLMM reports one **likelihood-ratio test per factor and
interaction** above its coefficients (`result.logistic.factor_tests`) — and these stay
valid when a factor separates pass from fail, where the per-coefficient p-values don't.
Each reduced model is refit with several optimizers until one passes lme4's checks; a term
where none does gets no test and a note. The statsmodels fallback (no R) has no such tests.

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
