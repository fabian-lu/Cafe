# Concepts

CAFE has a small vocabulary. Once these click, everything else follows.

## The black box

CAFE treats your system under test as a function:

```
run(config, item) -> output
```

- `config` is a flat dict mapping each **factor** to a chosen **level**, e.g.
  `{"model": "large", "top_k": 8, "temperature": 0.2}`.
- `item` is one element of your evaluation set (a question, task, or prompt).
- `output` is whatever your system returns.

Internally your system can be anything — a straight RAG pipeline, a router with
branches, a model cascade, an agent loop, a call to a production HTTP service.
**CAFE does not model your topology.** This is what makes it general.

## Factors

A **factor** is a named axis you vary. There are two kinds:

- **Parameter factors** are pure data — `temperature`, `top_k`, a model id,
  a prompt string. Zero code; you just list the levels.
- **Technique factors** are behaviors — a reranker, a router, a verifier. You
  implement these once in your own system code; then they're a level you select.

Each factor has a **type** that shapes design generation and (later) statistics:

| Type | Example | Meaning |
|------|---------|---------|
| `categorical` | `reranker ∈ {none, cross_encoder}` | unordered choices |
| `ordinal` | `effort ∈ {low, med, high}` | ordered choices |
| `continuous` | `temperature`, `top_k` | numeric knobs |

## Configurations, runs, replication

- A **configuration** (a "cell") is one full assignment of a level to every factor.
- **Replication** repeats each (configuration × input) several times. This is how
  CAFE measures **run-to-run nondeterminism** — the thing that lets it say whether
  a difference between two configs is *real* or just noise.

## Designs

A **design** decides which configurations to actually run. CAFE starts simple and
scales up:

- **`single`** — one configuration. "Just evaluate my system as-is." The on-ramp,
  and the basis for regression testing over time.
- **`full_factorial`** — every combination of every factor's levels. No limit on
  the number of factors or levels; the cost is the product of the level counts.

**Fractional factorial** designs (which trade completeness for far fewer runs) are
available now via `design="fractional"` — see `examples/06_fractional_design.ipynb`.
Screening, optimal, and response-surface designs are on the roadmap.

## What you get back

Running a study yields [`Observation`][cafe.Observation] records — one per executed
cell — collected in a [`Results`][cafe.Results] object. Each observation carries the
config, the output, timing, any error, and metadata (such as cost) your system
reported. From there, CAFE's statistics layer attributes quality to factors and
tests significance.
