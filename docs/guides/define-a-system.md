# Define your system

Your system under test is the **black box** CAFE runs the experiment around. You
can supply it three ways as a black box (all equivalent), or build it from
**techniques** for a transparent, instrumented pipeline (see
[Composed systems](#composed-systems-techniques)).

## A plain function

The simplest form. Sync or async both work:

```python
async def my_system(config, item):
    model = config["model"]
    temperature = float(config.get("temperature", 0.0))
    # ... call your pipeline / model / service ...
    return answer_text
```

## A function returning rich output

If you return a mapping with an `"output"` key, the other keys are captured as
per-observation **metadata** — this is how you surface cost, latency, or
intermediate artifacts to CAFE:

```python
async def my_system(config, item):
    answer, usage = await call_pipeline(config, item)
    return {
        "output": answer,
        "cost_usd": usage.cost,
        "latency_s": usage.latency,
        # later: "retrieved_chunks": [...] for component-level evaluation
    }
```

Report `cost_usd` / `tokens` on every answer, including a genuine `0.0` for a free local
model. An answer that leaves them out counts as *unreported*, not free. `cafe.pareto`
averages only the answers that report a value, and leaves the objective out of the
frontier if some configuration never reports it.

## An object with a `run` method

Handy when your system needs setup (clients, indexes) held as state:

```python
class MySystem:
    def __init__(self, client):
        self.client = client

    async def run(self, config, item):
        return await self.client.answer(item, **config)

study = cafe.Study(name="...", system=MySystem(client), factors=[...], dataset=[...])
```

## Composed systems (techniques)

Instead of a black box, you can build the system from **techniques** — named units (a
retriever, a reranker, a model call) grouped under **stages** — on a **`cafe.Pipeline`** you
own, run through an instrumented context `ctx`. CAFE then sees *inside* the pipeline: it
records per-stage time, tokens, and cost, captures a trace of every run, and can swap each
stage as a factor. The pipeline owns its techniques (no global state; two pipelines never
interfere). See `examples/02_technique_mode.ipynb` for the full walkthrough.

```python
pipe = cafe.Pipeline()

@pipe.technique("rerank", "cross_encoder")
async def cross_encoder(ctx, chunks):
    return rerank(chunks)

@pipe.compose                                            # marks the system function
async def system(config, item, ctx):
    chunks = await ctx.run("retrieve", query=item["text"])
    chunks = await ctx.run("rerank", chunks=chunks)      # may be skipped — see below
    return await ctx.run("generate", query=item["text"], chunks=chunks)

study = cafe.Study(
    system=pipe,                                         # the pipeline is the system
    factors=[pipe.factor("retrieve"), pipe.factor("rerank"), pipe.factor("generate")],
    dataset=[...],
)
```

The compose function returns the answer, or (like any system) a dict
`{"output": answer, "cost_usd": ..., ...}`. A `cost_usd` / `tokens` reported there, e.g. for a
priced call made directly in the compose function rather than through `ctx.run`, is added to
the answer's totals on top of the stages' own. It belongs to no stage, so it is not in the
trace or in `cafe.stage_report`. Other keys are kept as metadata; `trace` is always CAFE's.

A technique's keyword arguments *with defaults* (`top_k=5`) are tunable **parameters** — vary
them with a `"stage.param"` factor, e.g. `cafe.Factor("retrieve.top_k", [3, 5])`. Arguments
*without* defaults (`query`) are runtime inputs you pass via `ctx.run`. (For a deployed
catalog, register techniques programmatically with `pipe.add(stage, name, fn)`.)

A stage with **several** techniques needs a factor to choose one (`pipe.factor("rerank")`). A
stage with a **single** technique is *fixed* — it needs no factor of its own; CAFE just runs it.

Within one run (one configuration on one input), a repeated `ctx.run` with the same stage,
technique, parameters and inputs returns the first result without running the technique again;
the trace marks the repeat `cached: True` and its cost and tokens count once. Nothing is shared
*between* runs: every configuration, input and replication starts with an empty cache, so a
stage several configurations have in common (the same retriever under two generators) runs, and
is paid for, once per configuration. Plan the budget per configuration, not per distinct stage.

### Skipping a stage — the `None` level

To ask *"does this stage help at all?"*, add a **`None`** level to the stage's factor.
`None` means **skip the stage by passing its single input straight through** — CAFE runs no
technique and returns that input unchanged:

```python
cafe.Factor("rerank", ["cross_encoder", None])   # rerank on vs. off
```

Because skipping passes the input through *as the output*, the `None` level only makes sense
for a **stage whose output is the same kind of thing as its input** — e.g. a reranker
(`chunks → chunks`) or a post-processor (`answer → answer`). It **cannot** express "off" for
a stage that **changes the data type**. A retriever turns a `query` (string) into `docs` (a
list); skipping it would hand the next stage the raw *query* where it expects *documents*.
For "no retrieval", write an explicit technique that returns the empty result instead:

```python
@pipe.technique("retrieve", "none")     # "no retrieval" — return no documents
async def retrieve_none(ctx, query):
    return []
```

If a stage takes **more than one input**, name the one to pass through:
`pipe.factor("rerank", none="chunks")`.

### Cost of a run

A step's cost is the LLM cost CAFE observes inside it (via `cafe.complete`), plus the
technique's fixed `cost_usd=`, plus anything the technique adds with `ctx.add_cost(usd)`, plus
a `"cost_usd"` in a returned `{"output": ..., "cost_usd": ...}` dict. Steps may run
concurrently (`await asyncio.gather(ctx.run("a", ...), ctx.run("b", ...))`) or nested (a
technique that itself calls `ctx.run`); each step keeps its own cost. `ctx.add_cost` called
in the compose function itself, outside any `ctx.run`, counts toward the answer's total
`cost_usd` but belongs to no stage, so it does not show up in `cafe.stage_report`.

## Reading the config

`config` is a flat dict of `factor -> chosen level`. Read whatever your factors
defined:

```python
async def my_system(config, item):
    if config["router"] == "cascade":
        return await cascade(item, threshold=config["escalation_threshold"])
    return await single_model(item, model=config["model"])
```

A categorical factor like `router ∈ {single, cascade}` becomes a branch in your
code; a continuous factor like `escalation_threshold` becomes a number you read.
CAFE just decides which combination to run — your code decides what that means.

## Inputs and ids

`inputs` is a list; each element is one `item`. If an element is a mapping with an
`"id"` key, that id is used for resumable checkpointing; otherwise the item's
position is used.

```python
inputs = [
    {"id": "q1", "text": "What is the capital of France?"},
    {"id": "q2", "text": "Why is the sky blue?"},
]
```

## Errors

If your system raises, CAFE records the error on that observation and **keeps
going** — one bad cell never aborts the whole study. Inspect failures via
`results.errors`.
