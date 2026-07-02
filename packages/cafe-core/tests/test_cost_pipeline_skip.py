"""Composed-mode additions: user cost, per-technique cost, ctx.add_cost, skip level,
pipeline view."""

import cafe
from cafe.llm import _price_from_override, set_model_cost
from cafe.techniques import registry
from cafe.techniques.composed import pipeline, stage_report


def test_set_model_cost_override_and_clear():
    set_model_cost("acme/model", per_1k_input=1.0, per_1k_output=2.0)
    # 1000 prompt tokens @ $1/1k + 500 completion @ $2/1k = 1.0 + 1.0
    assert _price_from_override("acme/model", 1000, 500) == 2.0
    set_model_cost("acme/model", per_1k_tokens=0.5)   # blended
    assert _price_from_override("acme/model", 1000, 1000) == 1.0
    set_model_cost("acme/model")                       # clear
    assert _price_from_override("acme/model", 1000, 1000) is None


def _setup_pipeline():
    registry.clear()

    @cafe.technique("retrieve", "bm25", cost_usd=0.002)  # fixed non-LLM cost
    async def bm25(ctx, query, top_k=5):
        return f"chunks({query},k={top_k})"

    @cafe.technique("rerank", "cross_encoder")
    async def ce(ctx, chunks):
        ctx.add_cost(0.001)  # variable cost added by the technique
        return chunks + "+reranked"

    @cafe.technique("generate", "dummy")
    async def gen(ctx, query, chunks):
        return f"answer[{chunks}]"

    async def system(config, item, ctx):
        c = await ctx.run("retrieve", query=item)
        c = await ctx.run("rerank", chunks=c)
        return await ctx.run("generate", query=item, chunks=c)

    return system


def test_per_technique_and_manual_cost_flow_to_stage_report():
    system = _setup_pipeline()
    study = cafe.Study(
        name="c", system=cafe.composed(system),
        factors=[cafe.Factor("retrieve", ["bm25"]), cafe.Factor("rerank", ["cross_encoder"]),
                 cafe.Factor("generate", ["dummy"])],
        dataset=["q1", "q2"],
    )
    by_stage = {r["stage"]: r for r in stage_report(study.run())}
    assert by_stage["retrieve"]["mean_cost_usd"] == 0.002   # decorator cost
    assert by_stage["rerank"]["mean_cost_usd"] == 0.001     # ctx.add_cost
    assert by_stage["generate"]["mean_cost_usd"] == 0.0


def test_skip_level_passthrough():
    _setup_pipeline()
    # rerank turned off should pass the chunks straight through
    factor = cafe.technique_factor("rerank", none="chunks")
    assert "none" in factor.levels

    async def system(config, item, ctx):
        c = await ctx.run("retrieve", query=item)
        c = await ctx.run("rerank", chunks=c)   # may be skipped
        return c

    study = cafe.Study(
        name="s", system=cafe.composed(system),
        factors=[cafe.Factor("retrieve", ["bm25"]), factor],
        dataset=["q1"],
    )
    outs = {o.config["rerank"]: o.output for o in study.run()}
    assert outs["none"] == "chunks(q1,k=5)"                 # unchanged
    assert outs["cross_encoder"] == "chunks(q1,k=5)+reranked"


def test_none_level_via_plain_factor_skips():
    _setup_pipeline()

    async def system(config, item, ctx):
        c = await ctx.run("retrieve", query=item)
        return await ctx.run("rerank", chunks=c)   # skipped when rerank is None

    study = cafe.Study(
        name="s", system=cafe.composed(system),
        factors=[cafe.Factor("retrieve", ["bm25"]), cafe.Factor("rerank", ["cross_encoder", None])],
        dataset=["q1"],
    )
    outs = {o.config["rerank"]: o.output for o in study.run()}
    assert outs[None] == "chunks(q1,k=5)"                     # None → passthrough
    assert outs["cross_encoder"] == "chunks(q1,k=5)+reranked"


def test_none_level_renders_as_none_and_survives_stats():
    import cafe
    from cafe.judging.ratings import Rating, Ratings

    items = []
    for q in range(6):
        for rr in ["by_length", None]:            # None = skipped stage
            s = 5 if rr == "by_length" else 3
            items.append(Rating(obs_key=f"{rr}{q}", config={"rerank": rr}, input_id=f"q{q}",
                                rep=0, judge_rep=0, value=s, value_numeric=s))
    r = Ratings(rubric=cafe.ANSWER_QUALITY_1_5, judge_model="j", factors=["rerank"], items=items)
    levels = {m["level"] for m in cafe.attribute(r).factor_marginals}
    assert "none" in levels and None not in levels          # rendered as the string "none"
    assert cafe.fit_effects(r, interactions=1).n_obs == 12  # None rows kept, not dropped as NaN


def test_pipeline_view_order_and_levels():
    system = _setup_pipeline()
    study = cafe.Study(
        name="p", system=cafe.composed(system),
        factors=[cafe.Factor("retrieve", ["bm25"]),
                 cafe.Factor("rerank", ["cross_encoder"]),
                 cafe.Factor("generate", ["dummy"])],
        dataset=["q1"],
    )
    pl = pipeline(study)
    assert pl.order == ["retrieve", "rerank", "generate"]   # observed execution order
    assert "retrieve" in pl.show() and "→" in pl.show()
