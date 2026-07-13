# Quickstart

## Install

Requires **Python ≥ 3.11** and **R** — the mixed-effects models (ordinal CLMM, logistic GLMM) run in R.

```bash
git clone https://github.com/fabian-lu/Cafe.git
cd Cafe
pip install -e "packages/cafe-core[all]"    # library + LLM judge + datasets + notebooks

# R + the model packages  (Debian/Ubuntu shown; macOS: `brew install r`)
sudo apt install r-base
Rscript -e 'install.packages(c("ordinal", "lme4"))'
```

Verify the environment before your first run:

```bash
cafe doctor    # checks Python, the R stats engine, and LLM access
```

## Run the bundled example (no API keys)

CAFE ships a neutral toy system so you can see a full study run with nothing to
configure:

```bash
cafe run example            # 4 configs x 4 inputs x 3 reps
cafe run example --smoke    # preflight: 1 input, 1 rep, + cost/time estimate
cafe validate example       # expand the design without running anything
```

You'll see a per-configuration table with replication counts, latency, cost, and
(for the toy system) a simulated quality signal.

## Your first real study

A study is your **system** (a black box), the **factors** to vary, and the
**inputs** to evaluate on:

```python
import cafe

async def my_system(config, item):
    # Your compound system: RAG, routing, a cascade, an agent — anything.
    # Read the chosen levels from `config` and do whatever you want.
    model = config["model"]
    return f"[{model}] answer to: {item}"

study = cafe.Study(
    name="my-first-study",
    system=my_system,
    factors=[
        cafe.Factor("model", ["small", "large"]),
        cafe.Factor("prompt", ["plain", "cot"]),
    ],
    dataset=["What is 2+2?", "Summarize relativity."],
    replications=3,                       # measure run-to-run nondeterminism
)

results = study.run(checkpoint_path=".cafe/my-first-study.jsonl")
print(results.summary())
for obs in results:
    print(obs.config, "->", obs.output, f"({obs.elapsed_s}s)")
```

`study.run()` returns a [`Results`][cafe.Results] object you keep in a variable,
like any normal value. The optional `checkpoint_path` makes a long run
**crash-safe**: if it dies halfway, re-running **resumes** instead of restarting.

## From answers to attribution

`run()` gives you the raw answers. To get the actual payoff — *which factor drives
quality, and is the difference real* — add a **rubric** and a **judge**, then call
`evaluate()` instead:

```python
study = cafe.Study(
    name="my-first-study",
    system=my_system,
    factors=[cafe.Factor("model", ["small", "large"]),
             cafe.Factor("prompt", ["plain", "cot"])],
    dataset=["What is 2+2?", "Summarize relativity."],
    rubric=cafe.rubrics.CORRECTNESS_0_3,                 # the scale to grade on
    judge=cafe.LLMJudge(model="ollama_cloud/gpt-oss:20b"),
    replications=3,
)

result = study.evaluate()        # answers → judge → attribution
print(result.report())           # descriptive means, significance, effect sizes, the CLMM
```

`report()` prints per-factor significance (F-tests, p-values, partial η²), effect
sizes, the best configuration, and — for ordinal rubrics — the cumulative-link mixed
model. **[Interpreting results](interpreting-results.md)** walks through that output
line by line.

## Headless / CI

The same study runs from a Python file via the CLI:

```bash
cafe run path/to/study.py --checkpoint .cafe/run.jsonl --out results.jsonl
```

where `study.py` defines a module-level `study` (a `cafe.Study`) or a
`build_study()` function returning one.
