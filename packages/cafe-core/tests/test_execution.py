import asyncio


from cafe import Factor, Study, preflight, run_study
from cafe.execution import Checkpoint
from cafe.evaluation import Preflight
from cafe.execution.results import Observation, Results
from cafe.execution.runner import estimate


def _counting_study(fail_on=None):
    calls = {"n": 0}

    async def system(config, item):
        calls["n"] += 1
        if fail_on is not None and config["model"] == fail_on:
            raise RuntimeError("boom")
        return {"output": f"{config['model']}:{item}", "cost_usd": 0.01}

    study = Study(
        name="exec",
        system=system,
        factors=[Factor("model", ["a", "b"])],
        dataset=["q1", "q2"],
        replications=2,
    )
    return study, calls


async def test_runs_every_cell():
    study, calls = _counting_study()
    results = await run_study(study, concurrency=4)
    # 2 configs x 2 inputs x 2 reps
    assert len(results) == 8
    assert calls["n"] == 8
    assert results.summary()["n_errors"] == 0
    assert results.summary()["n_configs"] == 2


async def test_per_item_error_isolation():
    study, _ = _counting_study(fail_on="a")
    results = await run_study(study, concurrency=4)
    # config 'a' (4 cells) all error; 'b' (4 cells) all ok — run does not abort
    assert len(results) == 8
    assert len(results.errors) == 4
    assert all("RuntimeError" in o.error for o in results.errors)
    assert all(o.ok for o in results.observations if o.config["model"] == "b")


async def test_smoke_runs_one_input_one_rep():
    study, _ = _counting_study()
    results = await run_study(study, smoke=True)
    # 2 configs x 1 input x 1 rep
    assert len(results) == 2
    assert {o.input_id for o in results.observations} == {"in0"}


def _obs(elapsed_s, metadata):
    return Observation(
        config={"m": "a"}, input_id="q0", rep=0, output="x", elapsed_s=elapsed_s, metadata=metadata
    )


def test_estimate_keeps_measured_zero_cost_and_time():
    # A 0.0 mean (free / cached / local model) is a measurement, not "unavailable".
    results = Results(study_name="t", factors=["m"], observations=[_obs(0.0, {"cost_usd": 0.0})])
    est = estimate(results, total_cells=10)
    assert est["est_total_cost_usd"] == 0.0
    assert est["est_total_compute_s"] == 0.0
    text = Preflight(answers=results, estimate=est).show()
    assert "est. compute ~0.0s, est. cost ~$0.0" in text


def test_estimate_treats_none_cost_as_not_reported():
    obs = [_obs(1.0, {"cost_usd": None}), _obs(1.0, {"cost_usd": 0.02}), _obs(1.0, {})]
    est = estimate(Results(study_name="t", factors=["m"], observations=obs), total_cells=10)
    assert est["est_total_cost_usd"] == 0.2  # mean over the one cell that reported a cost
    assert est["est_total_compute_s"] == 10.0
    only_none = Results(study_name="t", factors=["m"], observations=[_obs(1.0, {"cost_usd": None})])
    assert estimate(only_none, total_cells=10)["est_total_cost_usd"] is None


async def test_checkpoint_resume_skips_done(tmp_path):
    cp_path = str(tmp_path / "run.jsonl")
    study, calls = _counting_study()

    # First run completes fully and writes the checkpoint.
    await run_study(study, checkpoint_path=cp_path)
    assert calls["n"] == 8
    assert len(Checkpoint(cp_path).load()) == 8

    # Second run with the same checkpoint should execute zero new cells.
    calls["n"] = 0
    results2 = await run_study(study, checkpoint_path=cp_path, resume=True)
    assert calls["n"] == 0
    assert len(results2) == 8


async def test_concurrency_is_bounded():
    in_flight = {"now": 0, "max": 0}

    async def system(config, item):
        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        await asyncio.sleep(0.01)
        in_flight["now"] -= 1
        return "x"

    study = Study(
        name="conc",
        system=system,
        factors=[Factor("x", list(range(10)))],
        dataset=["q"],
        replications=2,
    )
    await run_study(study, concurrency=3)
    assert in_flight["max"] <= 3


def test_sync_run_helper():
    study, _ = _counting_study()
    results = study.run()  # sync wrapper around asyncio.run
    assert len(results) == 8


async def test_sync_run_inside_event_loop():
    # This test body runs inside a running loop (like a Jupyter cell). The sync
    # study.run() must fall back to a worker thread instead of raising.
    study, _ = _counting_study()
    results = study.run()
    assert len(results) == 8
    assert results.summary()["n_errors"] == 0


async def test_checkpoint_resume_retries_errored_cells(tmp_path):
    cp_path = str(tmp_path / "run.jsonl")

    flaky = {"fail": True}
    calls = {"n": 0}

    async def system(config, item):
        calls["n"] += 1
        if flaky["fail"] and config["model"] == "b":
            raise RuntimeError("transient boom")
        return {"output": f"{config['model']}:{item}"}

    study = Study(
        name="exec-retry",
        system=system,
        factors=[Factor("model", ["a", "b"])],
        dataset=["q1", "q2"],
        replications=2,
    )

    # First run: every 'b' cell errors (4 of 8).
    r1 = await run_study(study, checkpoint_path=cp_path)
    assert len(r1.errors) == 4 and calls["n"] == 8

    # Resume with the provider healed: only the 4 errored cells re-run, all succeed.
    flaky["fail"] = False
    calls["n"] = 0
    r2 = await run_study(study, checkpoint_path=cp_path, resume=True)
    assert calls["n"] == 4
    assert len(r2) == 8 and len(r2.errors) == 0

    # retry_errors=False keeps errored cells as done.
    flaky["fail"] = True
    calls["n"] = 0
    r3 = await run_study(study, checkpoint_path=cp_path, resume=True, retry_errors=False)
    assert calls["n"] == 0 and len(r3) == 8


async def _preflight_text(outputs):
    async def system(config, item):
        if config["kind"] == "boom":
            raise RuntimeError("boom")
        return {"output": outputs[config["kind"]], "cost_usd": 0.01}

    study = Study(
        name="pf",
        system=system,
        factors=[Factor("kind", [*outputs, "boom"])],
        dataset=["q1"],
    )
    return (await preflight(study)).show()


async def test_preflight_show_renders_non_string_outputs():
    # A system's output may be any value, not just a str.
    text = await _preflight_text({"dict": {"answer": 42}, "list": ["a", "b"], "int": 7})
    assert "[kind=dict] {'answer': 42}" in text
    assert "[kind=list] ['a', 'b']" in text
    assert "[kind=int] 7" in text
    assert "[kind=boom] RuntimeError: boom" in text  # no output -> the error is shown


async def test_preflight_show_keeps_falsy_outputs():
    # 0 / False are answers; they must not be replaced by the (empty) error text.
    text = await _preflight_text({"zero": 0, "false": False})
    assert "[kind=zero] 0" in text
    assert "[kind=false] False" in text
    assert "[kind=boom] RuntimeError: boom" in text
