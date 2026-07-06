<p align="center">
  <img src="logo.png" alt="CAFE — Compound-AI Factorial Evaluation" width="420">
</p>

<h3 align="center">Stop guessing which config is better. Prove it.</h3>

<p align="center">
  A design-of-experiments platform for evaluating <b>compound AI systems</b>.
</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  <a href="https://fabian-lu.github.io/Cafe"><img alt="Docs" src="https://img.shields.io/badge/docs-online-success.svg"></a>
  <a href="https://arxiv.org/"><img alt="Paper" src="https://img.shields.io/badge/paper-arXiv-b31b1b.svg"></a>
</p>

<p align="center">
  <a href="https://cafe-ai.de/demo"><b>Live demo</b></a> &nbsp;·&nbsp;
  <a href="https://fabian-lu.github.io/Cafe">Documentation</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick start</a> &nbsp;·&nbsp;
  <a href="paper/cafe.pdf">Paper</a>
</p>

---

Modern AI applications are **compound systems**: pipelines of interacting techniques — retrieval,
reranking, context assembly, prompting, one or more model calls, tools, routers, verifiers. When the
output improves, *which part actually helped?* Aggregate benchmarks can't say.

**CAFE treats every pipeline knob as an experimental factor.** It generates factorial designs, runs each
configuration as a black box with replication, collects ordinal quality judgments from a configurable
LLM judge (and human experts), and attributes the variance in quality with mixed-effects and ordinal
models. You get a direct, statistically grounded answer to:

> **which technique drives quality, by how much, what the best configuration is — and whether the
> difference is real** given LLM run-to-run nondeterminism.

CAFE *measures* compound AI systems; it doesn't implement them. You bring the system as a black box —
CAFE runs the experiment.

<p align="center">
  <img src="assets/results.png" alt="CAFE Results dashboard: factor attribution, effect sizes, the ordinal model" width="900">
</p>

## Why CAFE

- **Attribution, not a leaderboard** — per-factor F-tests, p-values, and partial η² tell you which knob
  moves quality and by how much, holding the others fixed.
- **Statistically honest** — a linear mixed-effects model with a per-question random effect, plus the
  **scale-correct model** for your rubric (ordinal → cumulative-link mixed model, binary → logistic).
- **Built for LLM nondeterminism** — replication + significance testing separate real effects from
  run-to-run noise; a permutation-null keeps the false-positive rate honest.
- **Cost / quality trade-offs** — an automatic Pareto frontier over quality vs. cost, latency, tokens.
- **Human + LLM judges** — measure judge↔human agreement with Krippendorff's α (inter-rater reliability).
- **Efficient designs** — full and fractional factorial designs when the configuration space is large.
- **Beautiful, self-hostable UI** — a "Factorial Mono" web platform over the same engine; no data leaves
  your machine.

## Quick start

### The library

Requires **Python ≥ 3.11**.

```bash
git clone https://github.com/fabian-lu/Cafe.git
cd Cafe
pip install -e "packages/cafe-core[stats]"

cafe run example      # runs a bundled toy study — no API keys needed
cafe doctor           # checks optional prerequisites (R, for the ordinal/logistic models)
```

A minimal real study:

```python
import cafe

study = cafe.Study(
    name="rag-eval",
    system=pipe,                                 # your compound system as a black box
    factors=[
        pipe.factor("retrieve"),                 # e.g. none · dense · dense+rerank
        cafe.Factor("generate.model", ["small", "large"]),
    ],
    dataset=[{"text": "Who wrote Dune?", "reference": "Frank Herbert"}],
    rubric=cafe.rubrics.CORRECTNESS_0_3,         # a built-in ordinal rubric
    judge=cafe.LLMJudge(model="openai/gpt-4o-mini", preset="reference_qa"),
)

result = study.evaluate()      # generate answers → judge → attribute
print(result.report())         # significance, effect sizes, best config, the ordinal model
```

See the [notebooks](examples/) and the [documentation](https://fabian-lu.github.io/Cafe) for defining a
system, custom rubrics/judges, human ratings, and fractional designs.

### The platform (web app)

A self-hostable FastAPI + React platform over the same engine — set up a study, run it with live
progress, and explore the full analytics in the browser.

```bash
cd apps/web-app
cp .env.example .env      # add your LLM keys (never commit .env)
docker compose up
```

Open `http://localhost:5173`. Or just try the [**live demo**](https://cafe-ai.de/demo) (read-only).

## Repository layout

```
packages/cafe-core/   the evaluation engine — the pip-installable library + the `cafe` CLI
apps/web-app/         the self-hostable platform (FastAPI backend + React/Vite frontend)
apps/landing/         the landing page
techniques/           example systems-under-test — the extension point you copy and adapt
examples/             tutorial notebooks (also rendered into the docs)
docs/                 documentation source (MkDocs)
```

## How it works

1. **Declare factors.** Each pipeline technique or parameter you want to compare becomes a factor with a
   set of levels (retrieval ∈ {none, dense, rerank}; model ∈ {small, large}; …).
2. **Generate a design.** CAFE expands the full (or fractional) factorial — every combination to run.
3. **Execute as a black box.** Each configuration runs over your dataset with replication; execution is
   concurrent and crash-safe (resumable checkpoints).
4. **Judge.** A configurable LLM judge scores each answer on your rubric (reference-guided or
   reference-free); optionally collect human ratings too.
5. **Attribute.** Mixed-effects + scale-matched models give per-factor significance, effect sizes,
   variance explained, the best configuration, and the cost/quality frontier.

## Documentation

Full docs — guides, API reference, and runnable tutorials — at
**[fabian-lu.github.io/Cafe](https://fabian-lu.github.io/Cafe)**.

## Citation

If you use CAFE in your research, please cite:

```bibtex
@inproceedings{cafe2026,
  title     = {CAFE: A Design-of-Experiments Platform for Evaluating Compound AI Systems},
  author    = {Lukassen, Fabian and others},
  booktitle = {Proceedings of EMNLP 2026: System Demonstrations},
  year      = {2026}
}
```

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE).
