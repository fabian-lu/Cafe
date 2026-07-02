"""Run a coroutine to completion from sync code, in a script *or* a notebook.

If no event loop is running (a plain script) we just ``asyncio.run``. If one is already
running (e.g. a Jupyter cell), running another loop on the same thread would raise
``asyncio.run() cannot be called from a running event loop`` — so we run it on a short-lived
worker thread and join. This is what lets ``study.evaluate()`` / ``result.rejudge()`` be
plain sync calls everywhere.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable


def run_blocking(make_coro: Callable[[], Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coro())

    box: dict[str, Any] = {}

    def _worker() -> None:
        try:
            box["result"] = asyncio.run(make_coro())
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=_worker, name="cafe-run")
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["result"]
