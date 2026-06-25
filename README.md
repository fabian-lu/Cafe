<p align="center">
  <img src="logo.png" alt="CAFE — Compound AI Factorial Evaluation" width="440">
</p>

<p align="center"><b>A design-of-experiments platform for evaluating compound AI systems.</b></p>

<p align="center"><i>Stop guessing which config is better. Prove it.</i></p>

Modern AI applications are *compound systems*: pipelines of interacting techniques
(retrieval, reranking, prompting, one or more model calls, tools, routers,
verifiers). CAFE answers what aggregate benchmarks can't: **which technique drives
quality, by how much, what the best configuration is, and whether the difference is
real** given LLM run-to-run nondeterminism. It treats each pipeline knob as an
experimental **factor**, generates factorial designs, executes configurations with
replication, collects ordinal quality judgments (LLM judge + human experts), and
attributes variance with mixed-effects / ordinal models.

CAFE *measures* compound AI systems — it does not implement them. You bring the
system (as a black box); CAFE runs the experiment.

## Repository layout

```
CAFE/
├── packages/
│   ├── cafe-core/        # the evaluation engine (library + CLI)  ← built
│   └── cafe-techniques/  # neutral example techniques (extension template)
├── apps/
│   ├── backend/          # FastAPI + worker (web platform)
│   ├── frontend/         # React, "Factorial Mono" design
│   └── landing/          # marketing + docs + read-only demo
├── design/               # platform design docs + MVP scope
└── paper/                # the EMNLP 2026 demo paper
```

## Quick start

```bash
uv venv && uv pip install -e "packages/cafe-core[dev]"
cafe run example          # runs a toy 2-factor study, no API keys needed
```

See `design/01-platform-design.md` for the full design and `design/02-mvp-vs-future.md`
for what ships in the MVP.
