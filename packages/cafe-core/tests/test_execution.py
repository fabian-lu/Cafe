import asyncio


from cafe import Factor, Study, run_study
from cafe.checkpoint import Checkpoint


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
        inputs=["q1", "q2"],
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
        inputs=["q"],
        replications=2,
    )
    await run_study(study, concurrency=3)
    assert in_flight["max"] <= 3


def test_sync_run_helper():
    study, _ = _counting_study()
    results = study.run()  # sync wrapper around asyncio.run
    assert len(results) == 8
