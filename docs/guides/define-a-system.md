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

A technique's keyword arguments *with defaults* (`top_k=5`) are tunable **parameters** — vary
them with a `"stage.param"` factor, e.g. `cafe.Factor("retrieve.top_k", [3, 5])`. Arguments
*without* defaults (`query`) are runtime inputs you pass via `ctx.run`. (For a deployed
catalog, register techniques programmatically with `pipe.add(stage, name, fn)`.)

A stage with **several** techniques needs a factor to choose one (`pipe.factor("rerank")`). A
stage with a **single** technique is *fixed* — it needs no factor of its own; CAFE just runs it.

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
