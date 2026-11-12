"""Mode B (composed pipelines) framework — wiring tested offline (no LLM)."""

import asyncio

import pytest

import cafe
from cafe import llm
from cafe.techniques import stage_report


class _PrintsAlike:
    """Prints as "same" whatever its value, as two long numpy arrays can print alike
    ("[0. 0. 0. ... 0. 0. 0.]"). Module-level, so it can be pickled."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "same"


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


def test_compose_return_dict_cost_adds_to_run_total():
    # A compose fn may return {"output": ..., "cost_usd": ...} like any system (e.g. for a
    # priced call made outside ctx.run): that cost/tokens add to the answer's totals, the
    # per-stage trace stays as it was, and other keys are kept as metadata.
    pipe = cafe.Pipeline()

    @pipe.technique("retrieve", "bm25", cost_usd=0.01, energy_wh=0.5)
    async def bm25(ctx, query):
        return [query]

    @pipe.compose
    async def system(config, item, ctx):
        docs = await ctx.run("retrieve", query=item)
        return {"output": f"answer({docs})", "cost_usd": 0.05, "tokens": 120,
                "energy_wh": 0.25, "latency_s": 1.5, "trace": ["ignored"]}

    study = cafe.Study(name="t", system=pipe, factors=[cafe.Factor("x", ["a"])],
                       dataset=["q1", "q2"])
    res = study.run()
    assert sorted(o.output for o in res.observations) == ["answer(['q1'])", "answer(['q2'])"]
    for o in res.observations:
        assert o.metadata["cost_usd"] == 0.06        # 0.01 stage + 0.05 compose
        assert o.metadata["tokens"] == 120
        assert o.metadata["energy_wh"] == 0.75
        assert o.metadata["latency_s"] == 1.5
        assert [s["stage"] for s in o.metadata["trace"]] == ["retrieve"]
        assert o.metadata["trace"][0]["cost_usd"] == 0.01
    by_stage = {r["stage"]: r for r in stage_report(res)}
    assert by_stage["retrieve"]["mean_cost_usd"] == 0.01   # compose cost is in no stage

    import asyncio

    import pytest

    @pipe.compose
    async def misshaped(config, item, ctx):   # no "output" key: cost would be lost
        return {"answer": "x", "cost_usd": 0.05}

    with pytest.warns(UserWarning, match="without an 'output' key"):
        asyncio.run(pipe.run({}, "q1"))


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


def test_stage_cache_is_per_run_not_across_configurations():
    # Documented behaviour: an identical ctx.run repeated within ONE run is served from
    # the cache (counted once); a stage shared by several configurations is NOT shared —
    # every (config, item, rep) cell starts with an empty cache and runs it itself.
    calls = {"retrieve": 0}
    pipe = cafe.Pipeline()

    @pipe.technique("retrieve", "bm25", cost_usd=0.01)   # fixed stage, same inputs
    async def bm25(ctx, query):
        calls["retrieve"] += 1
        return f"docs({query})"

    @pipe.technique("generate", "small")
    async def small(ctx, docs):
        return f"small[{docs}]"

    @pipe.technique("generate", "large")
    async def large(ctx, docs):
        return f"large[{docs}]"

    @pipe.compose
    async def system(config, item, ctx):
        docs = await ctx.run("retrieve", query=item)
        again = await ctx.run("retrieve", query=item)   # identical call → cache hit
        assert again == docs
        return await ctx.run("generate", docs=docs)

    study = cafe.Study(
        name="t", system=pipe, factors=[pipe.factor("generate")],
        dataset=["q1", "q2"], replications=2,
    )
    res = study.run()
    assert len(res) == 8                        # 2 configs x 2 items x 2 reps
    assert calls["retrieve"] == 8               # once per cell, never across cells
    for o in res.observations:
        retrieves = [s for s in o.metadata["trace"] if s["stage"] == "retrieve"]
        assert [s["cached"] for s in retrieves] == [False, True]
        assert o.metadata["cost_usd"] == 0.01   # the cached repeat is not charged again
    assert {r["stage"]: r["calls"] for r in stage_report(res)}["retrieve"] == 8


def _sampler_pipe(calls):
    """One stochastic stage: every run returns a new sample and costs 0.01 USD, 10 tokens
    and 0.5 Wh."""
    pipe = cafe.Pipeline()

    @pipe.technique("generate", "sample", cost_usd=0.01, energy_wh=0.5)
    async def sample(ctx, query):
        calls.append(query)
        llm._usage_sink.get().append({"model": "m", "tokens": 10, "cost_usd": 0.0})
        return f"{query}#{len(calls)}"

    return pipe


def test_cache_false_runs_every_call_and_counts_each_cost():
    # Best-of-n / self-consistency: n identical calls with cache=False draw n samples, and
    # each one is traced and paid for.
    calls = []
    pipe = _sampler_pipe(calls)

    @pipe.compose
    async def best_of_3(config, item, ctx):
        return [await ctx.run("generate", query=item, cache=False) for _ in range(3)]

    out = asyncio.run(pipe.run({}, "q"))
    assert out["output"] == ["q#1", "q#2", "q#3"]
    assert calls == ["q", "q", "q"]
    assert [s["cached"] for s in out["trace"]] == [False, False, False]
    assert out["cost_usd"] == 0.03
    assert out["tokens"] == 30
    assert out["energy_wh"] == 1.5

    res = cafe.Study(name="t", system=pipe, factors=[cafe.Factor("x", ["a"])],
                     dataset=["q1", "q2"]).run()
    assert {r["stage"]: r["calls"] for r in stage_report(res)}["generate"] == 6


def test_uncached_call_neither_reads_nor_fills_the_cache():
    calls = []
    pipe = _sampler_pipe(calls)

    @pipe.compose
    async def cached_first(config, item, ctx):
        first = await ctx.run("generate", query=item)                # runs, fills the cache
        sample = await ctx.run("generate", query=item, cache=False)  # runs again
        again = await ctx.run("generate", query=item)                # served from `first`
        return [first, sample, again]

    out = asyncio.run(pipe.run({}, "q"))
    assert out["output"] == ["q#1", "q#2", "q#1"]
    assert [s["cached"] for s in out["trace"]] == [False, False, True]
    assert out["cost_usd"] == 0.02 and out["tokens"] == 20 and out["energy_wh"] == 1.0

    calls.clear()

    @pipe.compose
    async def uncached_first(config, item, ctx):
        sample = await ctx.run("generate", query=item, cache=False)  # runs, stores nothing
        first = await ctx.run("generate", query=item)                # runs: not served the sample
        again = await ctx.run("generate", query=item)                # served from `first`
        return [sample, first, again]

    out = asyncio.run(pipe.run({}, "q"))
    assert out["output"] == ["q#1", "q#2", "q#2"]
    assert [s["cached"] for s in out["trace"]] == [False, False, True]
    assert out["cost_usd"] == 0.02


def test_cache_is_ctx_runs_own_keyword_not_a_technique_input():
    pipe = cafe.Pipeline()

    # A technique input can't be named like a ctx.run keyword: it could never be passed.
    with pytest.raises(ValueError, match="input named 'cache'"):
        @pipe.technique("lookup", "kv")
        async def kv_input(ctx, key, cache):
            return key

    async def stage_input(ctx, stage):
        return stage

    with pytest.raises(ValueError, match="input named 'stage'"):
        pipe.add("lookup", "kv", stage_input)
    assert pipe.names_for("lookup") == []

    # A tunable param (it has a default) may be called cache: it comes from the config.
    @pipe.technique("lookup", "kv")
    async def kv(ctx, key, cache="warm"):
        return f"{key}:{cache}"

    @pipe.compose
    async def system(config, item, ctx):
        value = await ctx.run("lookup", key=item, cache=False)
        # cache= is not an input either, so a stage skipped by a None level still sees
        # exactly one input to pass through.
        return await ctx.run("rerank", text=value, cache=False)

    out = asyncio.run(pipe.run({"lookup.cache": "cold", "rerank": None}, "k"))
    assert out["output"] == "k:cold"

    @pipe.compose
    async def misuse(config, item, ctx):
        return await ctx.run("lookup", key=item, cache="yes")

    with pytest.raises(TypeError, match="True or False"):
        asyncio.run(pipe.run({}, "k"))


def _echo_pipe(calls):
    pipe = cafe.Pipeline()

    @pipe.technique("echo", "same")
    async def echo(ctx, x):
        calls.append(x)
        return x

    return pipe


def _echo_twice(first, second):
    """Run the echo stage on ``first``, then on ``second``, in one run (cache on)."""
    calls = []
    pipe = _echo_pipe(calls)

    @pipe.compose
    async def system(config, item, ctx):
        return [await ctx.run("echo", x=first), await ctx.run("echo", x=second)]

    out = asyncio.run(pipe.run({}, None))
    return out["output"], [s["cached"] for s in out["trace"]], calls


def test_cache_never_serves_one_input_the_output_of_another():
    # Different inputs that print the same (str() / JSON-with-default=str) must not share
    # an entry: each call runs and returns its own result.
    a, b = _PrintsAlike(1), _PrintsAlike(2)
    outs, cached, _ = _echo_twice(a, b)
    assert [o.value for o in outs] == [1, 2] and cached == [False, False]

    outs, cached, _ = _echo_twice("same", _PrintsAlike(3))   # a str, and an object printing it
    assert outs[0] == "same" and outs[1].value == 3 and cached == [False, False]

    for first, second in [((1, 2), [1, 2]),                   # json writes a tuple as a list
                          ({1: "a"}, {"1": "a"}),             # ... and an int key as a str
                          ({"x"}, "{'x'}")]:                  # default=str prints a set
        outs, cached, _ = _echo_twice(first, second)
        assert outs == [first, second] and cached == [False, False]
        assert [type(o) for o in outs] == [type(first), type(second)]

    class Unpicklable:                    # local, so it can't be pickled: never cached
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return "same"

    outs, cached, _ = _echo_twice(Unpicklable(1), Unpicklable(2))
    assert [o.value for o in outs] == [1, 2] and cached == [False, False]
    same = Unpicklable(3)
    outs, cached, calls = _echo_twice(same, same)  # a miss, never a wrong hit
    assert cached == [False, False] and len(calls) == 2


def test_cache_tells_numpy_arrays_that_print_alike_apart():
    np = pytest.importorskip("numpy")
    a = np.zeros(2000)
    b = np.zeros(2000)
    b[1000] = 1.0
    assert str(a) == str(b)           # both print as "[0. 0. 0. ... 0. 0. 0.]"
    outs, cached, _ = _echo_twice(a, b)
    assert outs[0] is a and outs[1] is b and cached == [False, False]

    outs, cached, calls = _echo_twice(a, a.copy())   # equal content: still one run
    assert cached == [False, True] and len(calls) == 1


def test_equal_inputs_still_share_the_cache():
    # Equal JSON values hit, however they were built (dict key order doesn't matter) ...
    first = {"query": "q", "filters": {"year": 2024, "lang": "en"}, "k": [1, 2.5, None, True]}
    second = {"k": [1, 2.5, None, True], "filters": {"lang": "en", "year": 2024}, "query": "q"}
    outs, cached, calls = _echo_twice(first, second)
    assert cached == [False, True] and len(calls) == 1 and outs[1] is outs[0]

    # ... and so do equal non-JSON values (same type, same content).
    for first, second in [((1, 2), (1, 2)), (_PrintsAlike(4), _PrintsAlike(4))]:
        outs, cached, calls = _echo_twice(first, second)
        assert cached == [False, True] and len(calls) == 1
