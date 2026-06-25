"""Progress bars for long studies, via tqdm (notebook / terminal / CLI aware).

``tqdm.auto`` renders an HTML bar in Jupyter and a text bar in a terminal. If tqdm
isn't installed the bar silently no-ops — progress is a convenience, never a
requirement. Execution code drives it through the same ``on_progress(item, done,
total)`` callback used for custom reporting.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

Callback = Callable[[Any, int, int], None]


@contextmanager
def progress_bar(total: int, desc: str, *, enabled: bool = True) -> Iterator[Callback | None]:
    """Yield an ``on_progress(item, done, total)`` callback backed by a tqdm bar.

    Yields ``None`` when disabled or tqdm is unavailable, so callers can do
    ``cb = on_progress or bar`` and guard with ``if cb``.
    """
    if not enabled:
        yield None
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield None
        return

    bar = tqdm(total=total, desc=desc)

    def update(_item: Any, done: int, _total: int) -> None:
        bar.n = done
        bar.refresh()

    try:
        yield update
    finally:
        bar.n = total
        bar.refresh()
        bar.close()
