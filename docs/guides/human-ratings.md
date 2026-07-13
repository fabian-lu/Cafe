# Human ratings & reliability

CAFE's whole analysis rests on the judge's scores — so the honest question is *does the
judge agree with people?* This guide covers collecting human ratings and measuring
inter-rater reliability (IRR), and running the full statistics on humans instead of the
judge.

## The workflow

1. **Export an answer sheet** — a CSV of answers for your experts to score, blind to
   the judge's verdicts.
2. **Experts fill in scores** — one column per rater.
3. **Read it back** and measure agreement.

```python
result = study.evaluate()          # you already have a judged study

# 1 · export a blank rating sheet (one column per rater)
cafe.answer_sheet(result, "sheet.csv", raters=("expert_1", "expert_2"))

# 2 · experts fill in the score columns in sheet.csv …

# 3 · read the filled sheet and measure agreement
human = cafe.human_ratings("sheet.csv")
rel = cafe.reliability(result, human=human)
print(rel.show())
```

A partially filled sheet is fine — rows with a blank score are skipped, so raters can
score a stratified sample rather than everything.

## Reliability

`cafe.reliability` computes **Krippendorff's α** with the disagreement metric matched
to your rubric's scale (ordinal, nominal, or interval). It covers the three comparisons
you actually want:

```python
cafe.reliability(result, human=human)                       # judge ↔ humans
cafe.reliability(human=human)                               # humans ↔ each other
cafe.reliability(raters={"120b": result, "20b": other})     # judge ↔ judge
```

α ≥ 0.80 is the conventional "reliable" band; 0.67–0.80 is tentative. If the judge
agrees with your experts about as well as they agree with each other, the judge is a
trustworthy stand-in for the rest of the study.

## Analyze the humans, not the judge

Sometimes you want the full attribution — which factor drove quality — computed on
**human** scores, to compare against the judge's story. `human_evaluation` returns an
`Evaluation` backed by the human ratings, so everything on
[Interpreting results](../interpreting-results.md) works on it:

```python
human_result = cafe.human_evaluation(result, human)
print(human_result.report())        # the same report — but from the humans
```

Several raters on one answer act like judge replications (averaged before the factor
models). For just the scores as a ratings object (to feed the stats functions directly)
use `cafe.ratings_from_human(result, human)`.

## Lower-level α

If you already have scores in hand — from any source — score a `{rater: {unit: value}}`
table directly:

```python
alpha = cafe.krippendorff_alpha(
    {"a": {"q1": 2, "q2": 1}, "b": {"q1": 2, "q2": 0}},
    metric="ordinal",
)
```

The full walkthrough — exporting sheets, ingesting them, and reading α — is in the
[Human ratings & IRR notebook](../notebooks/03_human_and_irr.ipynb).
