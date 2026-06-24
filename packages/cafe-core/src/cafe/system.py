"""The black box: how CAFE calls the system under test.

CAFE treats the system as a function ``run(config, item) -> output``. The user
can supply:

- a plain function (sync or async), or
- any object with a ``run(config, item)`` method (sync or async).

Either is normalized to an async ``System`` so the executor has one calling
convention. The output can be:

- a plain value (becomes the observation's ``output``), or
- a mapping with an ``"output"`` key; remaining keys (e.g. ``cost_usd``,
  ``latency_s``, ``artifacts``) are captured as per-observation metadata. This is
  the seam through which Mode-B instrumentation will flow.
"""

from __future__ import annotations

import inspect
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class System(Protocol):
    async def run(self, config: dict[str, Any], item: Any) -> Any: ...


class _CallableSystem:
    """Adapts a plain callable into an async System."""

    def __init__(self, fn: Any, name: str | None = None) -> None:
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "system")

    async def run(self, config: dict[str, Any], item: Any) -> Any:
        result = self._fn(config, item)
        if inspect.isawaitable(result):
            result = await result
        return result


class _MethodSystem:
    """Adapts an object exposing ``run(config, item)`` into an async System."""

    def __init__(self, obj: Any) -> None:
        self._obj = obj
        self.name = getattr(obj, "name", type(obj).__name__)

    async def run(self, config: dict[str, Any], item: Any) -> Any:
        result = self._obj.run(config, item)
        if inspect.isawaitable(result):
            result = await result
        return result


def as_system(obj: Any) -> System:
    """Normalize a user-supplied system into an async :class:`System`."""
    # An object with a .run method takes precedence over plain callability so a
    # class instance whose .run is the entry point works as expected.
    if hasattr(obj, "run") and callable(getattr(obj, "run")):
        return _MethodSystem(obj)
    if callable(obj):
        return _CallableSystem(obj)
    raise TypeError(
        "system must be callable as run(config, item) or expose a .run method; "
        f"got {type(obj).__name__}"
    )


def normalize_output(raw: Any) -> tuple[Any, dict[str, Any]]:
    """Split a system's return value into (output, metadata).

    A mapping with an ``"output"`` key is unpacked; its other keys become
    metadata. Anything else is treated as the output with empty metadata.
    """
    if isinstance(raw, dict) and "output" in raw:
        meta = {k: v for k, v in raw.items() if k != "output"}
        return raw["output"], meta
    return raw, {}
