"""Mode B (composed pipelines) framework — wiring tested offline (no LLM)."""

import cafe
from cafe.techniques import stage_report


def _pipe():
    pipe = cafe.Pipeline()

    @pipe.technique("answer", "short")
    async def short(ctx, q):
        return f"{q}=short"

    @pipe.technique("answer", "long")
    async def long(ctx, q, n=3):
        return {"output": f"{q}=long*{n}", "cost_usd": 0.01}

    @pipe.technique("refine", "none")
    async def none(ctx, text):
        return text

    @pipe.technique("refine", "polish")
    async def polish(ctx, text):
        return {"output": text + "+polished", "cost_usd": 0.02}

    @pipe.compose
    async def system(config, item, ctx):
        a = await ctx.run("answer", q=item)
        return await ctx.run("refine", text=a)

    return pipe


def test_auto_factor_levels_from_pipeline():
    pipe = _pipe()
    assert pipe.factor("answer").levels == ["short", "long"]
    assert pipe.factor("refine").levels == ["none", "polish"]


def test_composed_swaps_techniques_and_traces():
    pipe = _pipe()
    study = cafe.Study(
        name="t",
        system=pipe,
        factors=[pipe.factor("answer"), pipe.factor("refine")],
        dataset=["q1"],
    )
    res = study.run()
    assert len(res) == 4  # 2 answer x 2 refine
    outputs = {o.config["answer"] + "/" + o.config["refine"]: o.output for o in res}
    assert outputs["short/none"] == "q1=short"
    assert outputs["long/polish"] == "q1=long*3+polished"
    # every observation recorded a 2-stage trace
    for o in res.observations:
        stages = [s["stage"] for s in o.metadata["trace"]]
        assert stages == ["answer", "refine"]


def test_param_factor_overrides_default():
    pipe = _pipe()
    study = cafe.Study(
        name="t",
        system=pipe,
        factors=[
            cafe.Factor("answer", ["long"]),
            cafe.Factor("answer.n", [5]),       # override the technique's default n=3
            cafe.Factor("refine", ["none"]),
        ],
        dataset=["q1"],
    )
    res = study.run()
    assert res.observations[0].output == "q1=long*5"


def test_stage_report_aggregates_cost():
    pipe = _pipe()
    study = cafe.Study(
        name="t",
        system=pipe,
        factors=[cafe.Factor("answer", ["long"]), cafe.Factor("refine", ["polish"])],
        dataset=["q1", "q2"],
    )
    rows = stage_report(study.run())
    by_stage = {r["stage"]: r for r in rows}
    assert by_stage["answer"]["mean_cost_usd"] == 0.01
    assert by_stage["refine"]["mean_cost_usd"] == 0.02


def test_two_pipelines_do_not_share_state():
    p1 = _pipe()
    p2 = cafe.Pipeline()

    @p2.technique("answer", "only")
    async def only(ctx, q):
        return q

    # each pipeline sees ONLY its own techniques — no global leakage
    assert p1.names_for("answer") == ["short", "long"]
    assert p2.names_for("answer") == ["only"]


def test_single_technique_stage_needs_no_factor():
    pipe = cafe.Pipeline()

    @pipe.technique("answer", "a")
    async def a(ctx, q):
        return q

    @pipe.technique("answer", "b")
    async def b(ctx, q):
        return q.upper()

    @pipe.technique("format", "only")           # FIXED stage — a single technique
    async def fmt(ctx, x):
        return f"[{x}]"

    @pipe.compose
    async def system(config, item, ctx):
        r = await ctx.run("answer", q=item)     # varied (2 techniques) → needs a factor
        return await ctx.run("format", x=r)     # fixed (1 technique) → NO factor needed

    study = cafe.Study(name="t", system=pipe, factors=[pipe.factor("answer")], dataset=["q1"])
    outs = {o.config["answer"]: o.output for o in study.run()}
    assert outs["a"] == "[q1]"                  # format ran with no "format" factor
    assert outs["b"] == "[Q1]"


def test_ambiguous_stage_without_factor_errors():
    import asyncio

    import pytest

    pipe = cafe.Pipeline()

    @pipe.technique("answer", "a")
    async def a(ctx, q):
        return q

    @pipe.technique("answer", "b")
    async def b(ctx, q):
        return q

    @pipe.compose
    async def system(config, item, ctx):
        return await ctx.run("answer", q=item)   # 2 techniques, no factor → ambiguous

    with pytest.raises(KeyError, match="no factor to pick"):
        asyncio.run(pipe.run({}, "q1"))


def test_programmatic_add_matches_decorator():
    pipe = cafe.Pipeline()

    async def gen(ctx, q):
        return f"{q}!"

    pipe.add("answer", "prog", gen, cost_usd=0.003)
    assert pipe.names_for("answer") == ["prog"]
    assert pipe.get("answer", "prog").cost_usd == 0.003
