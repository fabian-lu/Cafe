"""A model without a known price: its calls cost "unknown" (None), not $0 — offline, with
a stand-in for litellm."""

import sys
import types
import warnings

import pytest

import cafe
from cafe import llm
from cafe.evaluation import Evaluation, Preflight
from cafe.execution.results import Observation, Results
from cafe.execution.runner import estimate
from cafe.judging.ratings import Rating, Ratings
from cafe.techniques import stage_report

# What the stand-in LiteLLM charges per call. Any other model raises, as
# litellm.completion_cost does for a model missing from its price map.
_LITELLM_PRICES = {"priced": 0.01, "local-free": 0.0}


class _Usage:
    prompt_tokens, completion_tokens, total_tokens = 100, 50, 150


def _resp(model):
    message = types.SimpleNamespace(content=f"answer from {model}")
    return types.SimpleNamespace(model=model, usage=_Usage(),
                                 choices=[types.SimpleNamespace(message=message)])


@pytest.fixture
def fake_litellm(monkeypatch):
    """litellm without the network: acompletion answers any model, completion_cost prices
    only the models in _LITELLM_PRICES. Also no price overrides, and no model warned yet."""
    fake = types.ModuleType("litellm")

    async def acompletion(**kwargs):
        return _resp(kwargs["model"])

    def completion_cost(completion_response):
        if completion_response.model not in _LITELLM_PRICES:
            raise Exception(f"This model isn't mapped yet. model={completion_response.model}")
        return _LITELLM_PRICES[completion_response.model]

    fake.acompletion = acompletion
    fake.completion_cost = completion_cost
    monkeypatch.setitem(sys.modules, "litellm", fake)
    monkeypatch.setattr(llm, "_model_prices", {})
    monkeypatch.setattr(llm, "_unpriced_warned", set())
    return fake


def _usage_of(*models):
    """The usage records of one completion per model, made inside a pipeline stage's sink."""
    sink: list = []
    token = llm._usage_sink.set(sink)
    try:
        for model in models:
            llm._emit_usage(model, _resp(model))
    finally:
        llm._usage_sink.reset(token)
    return sink


def test_unpriced_model_costs_unknown_and_warns_once_per_model(fake_litellm):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sink = _usage_of("unpriced", "unpriced", "also-unpriced")
    assert [u["cost_usd"] for u in sink] == [None, None, None]   # unknown, not 0.0
    assert [u["tokens"] for u in sink] == [150, 150, 150]        # tokens are still recorded
    messages = [str(w.message) for w in caught]
    assert len(messages) == 2                                    # once per model
    assert "no price known for model 'unpriced'" in messages[0]
    assert "cafe.set_model_cost('unpriced'" in messages[0]
    assert "'also-unpriced'" in messages[1]


def test_litellm_prices_a_genuine_zero_and_overrides_are_kept(fake_litellm):
    with warnings.catch_warnings():
        warnings.simplefilter("error")                        # none of these may warn
        assert [u["cost_usd"] for u in _usage_of("priced", "local-free")] == [0.01, 0.0]
        cafe.set_model_cost("unpriced", per_1k_input=0.1, per_1k_output=0.2)   # the remedy
        cafe.set_model_cost("priced", per_1k_tokens=1.0)      # beats LiteLLM's own price
        sink = _usage_of("unpriced", "priced")
    # 100 prompt + 50 completion tokens: 0.1 * 0.1 + 0.05 * 0.2, and 0.15 * 1.0
    assert [u["cost_usd"] for u in sink] == [0.02, 0.15]


async def test_complete_outside_a_stage_neither_prices_nor_warns(fake_litellm):
    # e.g. a judge or a black-box system: CAFE records no usage there, so nothing to price
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reply = await cafe.complete("unpriced", [{"role": "user", "content": "q"}])
    assert reply == "answer from unpriced"


def _pipe():
    pipe = cafe.Pipeline()

    @pipe.technique("retrieve", "metered", cost_usd=0.002)
    async def retrieve(ctx, q):
        return q

    @pipe.technique("generate", "llm", cost_usd=0.001)
    async def generate(ctx, q, model="priced"):
        draft = await cafe.complete("priced", [{"role": "user", "content": q}])
        return await cafe.complete(model, [{"role": "user", "content": draft}])

    @pipe.compose
    async def system(config, item, ctx):
        docs = await ctx.run("retrieve", q=item)
        return await ctx.run("generate", q=docs)

    return pipe


async def test_an_unpriced_call_makes_its_step_and_the_run_cost_unknown(fake_litellm):
    pipe = _pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        priced = await pipe.run({"generate.model": "priced"}, "q")
        unpriced = await pipe.run({"generate.model": "unpriced"}, "q")

    steps = {s["stage"]: s for s in priced["trace"]}
    assert steps["generate"]["cost_usd"] == 0.021            # 2 calls + the fixed 0.001
    assert priced["cost_usd"] == 0.023

    steps = {s["stage"]: s for s in unpriced["trace"]}
    assert steps["retrieve"]["cost_usd"] == 0.002             # other steps keep their cost
    assert steps["generate"]["cost_usd"] is None              # not the partial 0.011
    assert steps["generate"]["tokens"] == 300                 # both calls' tokens
    assert unpriced["cost_usd"] is None                       # not the partial 0.013
    assert unpriced["tokens"] == 300


async def test_costs_outside_a_stage_keep_an_unknown_total_unknown(fake_litellm):
    pipe = _pipe()

    @pipe.compose
    async def system(config, item, ctx):
        ctx.add_cost(0.05)                                    # before the unknown step
        docs = await ctx.run("retrieve", q=item)
        answer = await ctx.run("generate", q=docs)
        ctx.add_cost(0.05)                                    # after it
        return {"output": answer, "cost_usd": 0.5}            # spent outside any stage

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = await pipe.run({"generate.model": "unpriced"}, "q")
    assert out["cost_usd"] is None


def _mixed_study():
    return cafe.Study(name="s", system=_pipe(), dataset=["q1", "q2"],
                      factors=[cafe.Factor("generate.model", ["priced", "unpriced"])])


def test_stage_report_cost_is_unknown_if_any_of_its_calls_is(fake_litellm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = {r["stage"]: r for r in stage_report(_mixed_study().run())}
    assert rows["retrieve"]["mean_cost_usd"] == 0.002
    assert rows["generate"]["mean_cost_usd"] is None          # not a mean of the priced runs
    assert rows["generate"]["calls"] == 4
    assert rows["generate"]["mean_tokens"] == 300.0


def test_pareto_leaves_out_the_cost_of_an_unpriced_configuration(fake_litellm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        answers = _mixed_study().run()
    ratings = Ratings(
        rubric=cafe.ANSWER_QUALITY_1_5, judge_model="m", factors=["generate.model"],
        items=[Rating(obs_key=o.key(), config=o.config, input_id=o.input_id, rep=o.rep,
                      judge_rep=0, value=4, value_numeric=4) for o in answers.observations],
    )
    res = cafe.pareto(Evaluation(study_name="s", answers=answers, ratings=ratings))
    cost = {r["config"]["generate.model"]: r["cost"] for r in res.rows}
    assert cost == {"priced": 0.023, "unpriced": None}         # not a cheaper-looking 0.013
    assert "cost" not in res.objectives


def _obs(config, cost):
    return Observation(config={"m": config}, input_id="q0", rep=0, output="x", elapsed_s=1.0,
                       metadata={"cost_usd": cost})


def test_preflight_says_what_its_cost_estimate_rests_on():
    partial = Results(study_name="t", factors=["m"],
                      observations=[_obs("a", 0.02), _obs("b", None)])
    est = estimate(partial, total_cells=10)
    assert est["est_total_cost_usd"] == 0.2 and est["cost_known_cells"] == 1
    text = Preflight(answers=partial, estimate=est).show()
    assert "est. cost ~$0.2" in text
    assert "extrapolated from the 1 of 2 sampled answers with a known cost_usd" in text

    unknown = Results(study_name="t", factors=["m"], observations=[_obs("a", None)])
    est = estimate(unknown, total_cells=10)
    assert est["est_total_cost_usd"] is None and est["cost_known_cells"] == 0
    text = Preflight(answers=unknown, estimate=est).show()
    assert "est. cost unknown (no answer has a known cost_usd" in text
    assert "cafe.set_model_cost" in text and "extrapolated" not in text
