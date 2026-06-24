# CAFE

**Compound-AI Factorial Evaluation** — a design-of-experiments platform for
evaluating compound AI systems.

> Stop guessing which config is better. Prove it.

Modern AI applications are *compound systems*: pipelines of interacting techniques
— retrieval, reranking, prompting, one or more model calls, tools, routers,
verifiers. CAFE answers what aggregate benchmarks can't:

- **Which technique drives quality**, and by how much?
- **What is the best configuration?**
- **Is the difference real**, or just LLM run-to-run noise?

CAFE treats each pipeline knob as an experimental **factor**, generates factorial
designs, executes configurations with **replication**, collects ordinal quality
judgments (LLM judge + human experts), and attributes variance with mixed-effects
and ordinal models.

!!! note "CAFE measures; it does not implement"
    You bring your system as a **black box** — `run(config, item) -> output`. CAFE
    runs the experiment around it. It never needs to know your pipeline's topology,
    which is why it works for *any* compound system: RAG, routing, cascades, agents.

## Where to go next

- **[Quickstart](quickstart.md)** — run your first study in a few lines, no API keys.
- **[Concepts](concepts.md)** — black box, factors, designs, replication.
- **[Define your system](guides/define-a-system.md)** — wire your own system in.
- **[API reference](reference/api.md)** — the `cafe` library, generated from source.

## Status

CAFE is in active development toward an open-source release and an EMNLP 2026
System Demonstration. The evaluation engine (`cafe-core`) is the first component;
judging, statistics, the web platform, and the docs are landing incrementally.
