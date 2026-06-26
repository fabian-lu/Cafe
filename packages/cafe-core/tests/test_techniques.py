"""Mode B (composed pipelines) framework — wiring tested offline (no LLM)."""

import cafe
from cafe.techniques import registry, stage_report


def _setup():
    registry.clear()

    @cafe.technique("answer", "short")
    async def short(ctx, q):
        return f"{q}=short"

    @cafe.technique("answer", "long")
    async def long(ctx, q, n=3):
        return {"output": f"{q}=long*{n}", "cost_usd": 0.01}

    @cafe.technique("refine", "none")
    async def none(ctx, text):
        return text

    @cafe.technique("refine", "polish")
    async def polish(ctx, text):
        return {"output": text + "+polished", "cost_usd": 0.02}

    async def system(config, item, ctx):
        a = await ctx.run("answer", q=item)
        return await ctx.run("refine", text=a)

    return system


def test_auto_factor_levels_from_registry():
    _setup()
    assert cafe.technique_factor("answer").levels == ["short", "long"]
    assert cafe.technique_factor("refine").levels == ["none", "polish"]


def test_composed_swaps_techniques_and_traces():
    system = _setup()
    study = cafe.Study(
        name="t",
        system=cafe.composed(system),
        factors=[cafe.technique_factor("answer"), cafe.technique_factor("refine")],
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
    system = _setup()
    study = cafe.Study(
        name="t",
        system=cafe.composed(system),
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
    system = _setup()
    study = cafe.Study(
        name="t",
        system=cafe.composed(system),
        factors=[cafe.Factor("answer", ["long"]), cafe.Factor("refine", ["polish"])],
        dataset=["q1", "q2"],
    )
    rows = stage_report(study.run())
    by_stage = {r["stage"]: r for r in rows}
    assert by_stage["answer"]["mean_cost_usd"] == 0.01
    assert by_stage["refine"]["mean_cost_usd"] == 0.02
