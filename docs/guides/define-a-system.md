# Define your system

Your system under test is the **black box** CAFE runs the experiment around. You
can supply it three ways — all equivalent.

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
