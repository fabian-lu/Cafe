# cafe-core

The CAFE evaluation engine: define a compound AI system as a black box, declare
the factors to vary, and run a design-of-experiments study over it — headless,
no web stack, no database required.

> Status: **Slice 1** — the engine (design generation + robust async execution +
> resumable checkpoints + CLI). Judging and statistics arrive in later slices.

## Install (dev)

```bash
uv venv && uv pip install -e "packages/cafe-core[dev]"
```

## Try it (no API keys)

```bash
cafe run example            # run the bundled toy 2-factor study
cafe run example --smoke    # preflight: 1 input, 1 rep, cost/time estimate
cafe validate example       # expand the design without running
```

## Library use

```python
import cafe

async def my_system(config, item):
    # your compound system: routing, RAG, cascade, agent — anything.
    return f"answer for {item!r} using {config['model']}"

study = cafe.Study(
    name="my-study",
    system=my_system,
    factors=[
        cafe.Factor("model", ["small", "large"]),
        cafe.Factor("prompt", ["plain", "cot"]),
    ],
    inputs=["question 1", "question 2"],
    replications=3,
)

results = study.run(checkpoint_path=".cafe/my-study.jsonl")  # resumable
print(results.summary())
for obs in results:
    print(obs.config, obs.output, obs.elapsed_s)
```

`study.run()` returns a `Results` object you hold like any value. The optional
`checkpoint_path` makes a long run crash-safe: re-running resumes instead of
restarting.
