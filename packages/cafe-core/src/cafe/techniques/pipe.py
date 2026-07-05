"""``Pipeline`` — a composed system you build from techniques, scoped to itself.

Create a pipeline, register techniques onto it, mark the compose function, and hand
the pipeline itself to the study as its ``system``:

    pipe = cafe.Pipeline()

    @pipe.technique("retrieve", "dense")
    async def dense(ctx, query, top_k=4):
        ...

    @pipe.compose
    async def run(config, item, ctx):
        docs = await ctx.run("retrieve", query=item["text"])
        ...

    study = cafe.Study(system=pipe, factors=[pipe.factor("retrieve"), ...], ...)

Everything is owned by the instance: re-running a notebook cell rebuilds it cleanly
(no ``registry.clear()``), and two pipelines never share state. A deployed catalog can
build one per run via :meth:`add` — see :meth:`add`.
"""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable

from cafe.study import Factor
from cafe.techniques.context import Context
from cafe.techniques.registry import TechniqueSpec

ComposeFn = Callable[[dict, Any, Context], Awaitable[Any]]

_NONE_UNSET = object()


class Pipeline:
    """A scoped collection of techniques + a compose function, usable as a Study ``system``."""

    def __init__(self, compose: ComposeFn | None = None) -> None:
        self._techniques: dict[tuple[str, str], TechniqueSpec] = {}
        self._compose: ComposeFn | None = compose
        #: A label for this system (shown in results); set from the compose fn's name.
        self.name = getattr(compose, "__name__", "pipeline")

    # ── registering techniques ─────────────────────────────────────────────────
    def _spec(self, stage: str, name: str, fn: Callable, description: str, cost_usd: float) -> None:
        params = {
            p.name: p.default
            for p in inspect.signature(fn).parameters.values()
            if p.default is not inspect.Parameter.empty
        }
        # Redefining replaces — like re-`def`-ing a function; makes notebook re-runs painless.
        self._techniques[(stage, name)] = TechniqueSpec(stage, name, fn, params, description, cost_usd)

    def technique(
        self, stage: str, name: str, *, description: str = "", cost_usd: float = 0.0
    ) -> Callable:
        """Register ``fn`` as the technique ``name`` for ``stage`` (decorator).

        ``cost_usd`` is a fixed cost charged every time this technique runs — for a non-LLM
        component whose price CAFE can't see (a paid reranker, a web-search API, a human
        step). It's added to any LLM cost tracked inside the function and shows up in
        ``stage_report`` / Pareto / ``report()``.
        """
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._spec(stage, name, fn, description, cost_usd)
            return fn

        return deco

    def add(
        self, stage: str, name: str, fn: Callable, *, description: str = "", cost_usd: float = 0.0
    ) -> "Pipeline":
        """Register a technique programmatically (not as a decorator) — e.g. a deployed
        catalog assembling a pipeline per run. Returns ``self`` for chaining."""
        self._spec(stage, name, fn, description, cost_usd)
        return self

    def compose(self, fn: ComposeFn) -> ComposeFn:
        """Mark ``fn(config, item, ctx)`` as this pipeline's system function (decorator)."""
        self._compose = fn
        self.name = getattr(fn, "__name__", "pipeline")
        return fn

    # ── introspection ──────────────────────────────────────────────────────────
    def names_for(self, stage: str) -> list[str]:
        """Technique names registered under ``stage`` (registration order)."""
        return [n for (s, n) in self._techniques if s == stage]

    def stages(self) -> list[str]:
        """Stages that have at least one registered technique (registration order)."""
        seen: list[str] = []
        for s, _ in self._techniques:
            if s not in seen:
                seen.append(s)
        return seen

    def get(self, stage: str, name: str) -> TechniqueSpec:
        try:
            return self._techniques[(stage, name)]
        except KeyError:
            raise KeyError(
                f"no technique {name!r} for stage {stage!r}; have: {self.names_for(stage)}"
            ) from None

    # ── build a factor from this pipeline's techniques ─────────────────────────
    def factor(
        self,
        stage: str,
        names: list[str] | None = None,
        *,
        none: Any = _NONE_UNSET,
        none_name: str = "none",
        **factor_kwargs: Any,
    ) -> Factor:
        """A categorical factor whose levels are ``stage``'s registered techniques.

        ``pipe.factor("retrieve")`` → ``Factor("retrieve", ["dense", "rerank"])``.

        Pass ``none=`` to add a "skip this stage" level without a no-op technique — the honest
        "does this stage help at all?" contrast (only valid when the stage passes its input
        through, i.e. output type == input type):

        - ``none="chunks"`` → the skip level returns the ``chunks`` input unchanged;
        - ``none=None``     → the skip level contributes nothing / returns ``None``.
        """
        levels = list(names if names is not None else self.names_for(stage))
        if none is not _NONE_UNSET:
            self._register_passthrough(stage, none_name, None if none is None else str(none))
            if none_name not in levels:
                levels.append(none_name)
        if not levels:
            raise ValueError(f"no techniques registered for stage {stage!r} on this pipeline")
        return Factor(stage, levels, **factor_kwargs)

    def _register_passthrough(self, stage: str, name: str, returns: str | None) -> None:
        if (stage, name) in self._techniques:
            return

        async def _passthrough(ctx, **inputs):  # noqa: ANN001, ANN003
            if returns is None:
                return None
            if returns not in inputs:
                raise KeyError(
                    f"skip level {stage!r}/{name!r} passes through {returns!r}, but "
                    f"ctx.run({stage!r}, ...) got inputs {sorted(inputs)}"
                )
            return inputs[returns]

        self._techniques[(stage, name)] = TechniqueSpec(
            stage, name, _passthrough, {}, description=f"skip {stage} (passthrough {returns})"
        )

    # ── act as the Study's system (the System protocol: async run(config, item)) ─
    async def run(self, config: dict[str, Any], item: Any) -> dict[str, Any]:
        """Execute the compose function for one (config, item), returning the answer plus
        per-stage trace + total cost/tokens as metadata (same contract as any system)."""
        if self._compose is None:
            raise ValueError("this Pipeline has no compose function — decorate one with @pipe.compose")
        ctx = Context(self._techniques, config)
        result = await self._compose(config, item, ctx)
        output = result["output"] if isinstance(result, dict) and "output" in result else result
        return {
            "output": output,
            "cost_usd": round(ctx.total_cost, 6),
            "tokens": ctx.total_tokens,
            "trace": ctx.trace,
        }
