"""Run a coroutine to completion from sync code, in a script *or* a notebook.

If no event loop is running (a plain script) we just ``asyncio.run``. If one is already
running (e.g. a Jupyter cell), running another loop on the same thread would raise
``asyncio.run() cannot be called from a running event loop`` — so we run it on a short-lived
worker thread and wait for it. This is what lets ``study.evaluate()`` / ``result.rejudge()`` be
plain sync calls everywhere.

An interrupt stops the run either way. In a script, ``asyncio.run`` cancels it on Ctrl-C. In
a notebook, "interrupt kernel" raises ``KeyboardInterrupt`` in the thread that waits for the
worker; we then cancel the worker's run and wait for it to wind down before re-raising, so the
run makes no further calls and writes no further checkpoint rows once the cell has stopped.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable


def run_blocking(make_coro: Callable[[], Any]) -> Any:
    """Run the coroutine ``make_coro()`` to completion and return its result.

    Parameters
    ----------
    make_coro:
        Builds the coroutine to run. Called once, on the thread that runs it.

    Returns
    -------
    Any
        The coroutine's result.

    Raises
    ------
    BaseException
        Whatever the coroutine raises. If the caller is interrupted while the coroutine runs on
        the worker thread (``KeyboardInterrupt`` from "interrupt kernel"), the coroutine is
        cancelled, and the interrupt is re-raised once the worker's run has stopped.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coro())

    box: dict[str, Any] = {}
    started, finished = threading.Event(), threading.Event()

    async def _main() -> Any:
        # Expose the worker's loop and task so the caller's thread can cancel the run.
        box["loop"], box["task"] = asyncio.get_running_loop(), asyncio.current_task()
        started.set()
        return await make_coro()

    def _worker() -> None:
        try:
            box["result"] = asyncio.run(_main())
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller's thread
            box["error"] = exc
        finally:
            started.set()
            finished.set()

    threading.Thread(target=_worker, name="cafe-run").start()
    try:
        # Wait in short slices so the caller stays interruptible (one blocking wait can't be
        # interrupted on every platform). Not Thread.join(): before Python 3.13, an interrupt
        # inside join() can mark the thread as finished while it still runs.
        while not finished.wait(0.1):
            pass
    except BaseException:
        # Interrupted (e.g. Jupyter's "interrupt kernel"). Left alone, the run would carry on
        # in the background, making every remaining call and writing to the checkpoint, and
        # re-running the cell to resume would start a second run alongside it. So cancel it
        # and wait until it has stopped. Rows already checkpointed stay; a cell cancelled
        # mid-call isn't written, so a resumed run redoes it.
        started.wait()
        if "task" in box:
            try:
                box["loop"].call_soon_threadsafe(box["task"].cancel)
            except RuntimeError:  # the run finished and closed its loop meanwhile
                pass
        finished.wait()
        raise
    if "error" in box:
        raise box["error"]
    return box["result"]
