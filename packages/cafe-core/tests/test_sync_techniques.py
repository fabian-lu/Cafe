"""Techniques and compose functions written as a plain ``def`` instead of an ``async def``."""

import asyncio

import cafe


def test_plain_def_technique_runs_like_an_async_one():
    pipe = cafe.Pipeline()
    calls = []

    @pipe.technique("retrieve", "keyword", cost_usd=0.001)
    def keyword(ctx, query, top_k=3):
        calls.append(query)
        ctx.add_cost(0.0005)
        return query.split()[:top_k]

    async def _model(ctx, words):
        ctx.add_cost(0.002)  # awaited inside the step, so the cost is the step's
        return " ".join(words).upper()

    @pipe.technique("generate", "upper")
    def upper(ctx, words):  # a plain def handing back an awaitable is awaited, as before
        return _model(ctx, words)

    @pipe.compose
    async def system(config, item, ctx):
        words = await ctx.run("retrieve", query=item)
        again = await ctx.run("retrieve", query=item)  # identical call: served from the cache
        return await ctx.run("generate", words=words + again)

    out = asyncio.run(pipe.run({"retrieve.top_k": 2}, "a b c"))

    assert out["output"] == "A B A B"
    assert calls == ["a b c"]
    retrieve, cached, generate = out["trace"]
    assert retrieve["params"] == {"top_k": 2} and retrieve["cost_usd"] == 0.0015
    assert cached["cached"] is True
    assert generate["technique"] == "upper" and generate["cost_usd"] == 0.002
    assert out["cost_usd"] == 0.0035


def test_plain_def_compose_function():
    pipe = cafe.Pipeline()

    @pipe.technique("answer", "echo")
    async def echo(ctx, q):
        return q

    @pipe.compose
    def system(config, item, ctx):
        ctx.add_cost(0.25)
        return {"output": item.upper(), "tokens": 7}

    out = asyncio.run(pipe.run({}, "hi"))
    assert out["output"] == "HI"
    assert out["cost_usd"] == 0.25 and out["tokens"] == 7 and out["trace"] == []


def test_plain_def_techniques_in_a_study_have_no_errors():
    pipe = cafe.Pipeline()

    @pipe.technique("answer", "sync")
    def sync(ctx, q):
        return f"{q}=sync"

    @pipe.technique("answer", "async")
    async def async_(ctx, q):
        return f"{q}=async"

    @pipe.compose
    def system(config, item, ctx):
        return ctx.run("answer", q=item)  # a plain def may hand back ctx.run's awaitable

    study = cafe.Study(name="t", system=pipe, factors=[pipe.factor("answer")], dataset=["q1"])
    res = study.run()
    assert [o.error for o in res.observations] == [None, None]
    assert {o.config["answer"]: o.output for o in res} == {"sync": "q1=sync", "async": "q1=async"}
