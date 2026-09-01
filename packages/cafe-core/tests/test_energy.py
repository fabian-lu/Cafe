"""Opt-in energy declarations on techniques — wiring tested offline (no LLM)."""

import asyncio

import cafe
from cafe import llm
from cafe.techniques import stage_report


def _run(coro):
    return asyncio.run(coro)


def _energy_pipe():
    pipe = cafe.Pipeline()

    @pipe.technique("answer", "fixed", energy_wh=0.5)
    async def fixed(ctx, q):
        return f"{q}=fixed"

    @pipe.technique("answer", "scaled", energy_wh_per_1k_tokens=2.0)
    async def scaled(ctx, q):
        # Simulate what cafe.complete does: report token usage to the active sink.
        sink = llm._usage_sink.get()
        sink.append({"model": "m", "tokens": 500, "cost_usd": 0.0})
        return f"{q}=scaled"

    @pipe.technique("answer", "both", energy_wh=0.1, energy_wh_per_1k_tokens=1.0)
    async def both(ctx, q):
        sink = llm._usage_sink.get()
        sink.append({"model": "m", "tokens": 200, "cost_usd": 0.0})
        return f"{q}=both"

    @pipe.technique("answer", "undeclared")
    async def undeclared(ctx, q):
        return f"{q}=undeclared"

    @pipe.compose
    async def system(config, item, ctx):
        return await ctx.run("answer", q=item)

    return pipe


def test_fixed_energy_per_run():
    out = _run(_energy_pipe().run({"answer": "fixed"}, "q"))
    assert out["energy_wh"] == 0.5
    assert out["trace"][0]["energy_wh"] == 0.5


def test_scaled_energy_uses_observed_tokens():
    out = _run(_energy_pipe().run({"answer": "scaled"}, "q"))
    assert out["energy_wh"] == 1.0  # 500 tokens * 2.0 Wh / 1k
    assert out["tokens"] == 500


def test_fixed_plus_scaled_combine():
    out = _run(_energy_pipe().run({"answer": "both"}, "q"))
    assert out["energy_wh"] == 0.3  # 0.1 fixed + 200 * 1.0 / 1k


def test_undeclared_technique_has_no_energy_key():
    out = _run(_energy_pipe().run({"answer": "undeclared"}, "q"))
    assert "energy_wh" not in out
    assert all("energy_wh" not in step for step in out["trace"])


def test_skip_level_has_no_energy_key():
    pipe = cafe.Pipeline()

    @pipe.technique("answer", "fixed", energy_wh=0.5)
    async def fixed(ctx, q):
        return q

    @pipe.compose
    async def system(config, item, ctx):
        a = await ctx.run("answer", q=item)
        return await ctx.run("refine", text=a)

    pipe.factor("refine", none="text")  # registers the passthrough skip technique
    out = _run(pipe.run({"answer": "fixed", "refine": "none"}, "q"))
    # The declared technique contributes; the passthrough contributes nothing.
    assert out["energy_wh"] == 0.5
    refine_steps = [s for s in out["trace"] if s["stage"] == "refine"]
    assert refine_steps and all("energy_wh" not in s for s in refine_steps)


def test_cached_step_counts_energy_once():
    pipe = cafe.Pipeline()

    @pipe.technique("answer", "fixed", energy_wh=0.5)
    async def fixed(ctx, q):
        return q

    @pipe.compose
    async def system(config, item, ctx):
        await ctx.run("answer", q=item)   # second call is a cache hit
        return await ctx.run("answer", q=item)

    out = _run(pipe.run({"answer": "fixed"}, "q"))
    assert out["energy_wh"] == 0.5


def test_stage_report_energy_only_when_declared():
    class Obs:
        def __init__(self, trace):
            self.metadata = {"trace": trace}

    class Res:
        observations = [
            Obs([{"stage": "answer", "technique": "fixed", "elapsed_s": 1.0,
                  "cost_usd": 0.0, "tokens": 10, "cached": False, "energy_wh": 0.5}]),
            Obs([{"stage": "answer", "technique": "undeclared", "elapsed_s": 1.0,
                  "cost_usd": 0.0, "tokens": 10, "cached": False}]),
        ]

    rows = {r["technique"]: r for r in stage_report(Res())}
    assert rows["fixed"]["mean_energy_wh"] == 0.5
    assert "mean_energy_wh" not in rows["undeclared"]
