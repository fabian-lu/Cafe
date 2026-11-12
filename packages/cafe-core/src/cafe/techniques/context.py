"""The instrumented context handed to a composed system.

When your system calls ``await ctx.run("retriever", query=q)``, CAFE:
  - reads the active config to see which technique fills that stage,
  - resolves the technique's tunable params from the config,
  - runs it, **times it**, captures its **cost** and **output**,
  - appends a step to a **trace**, and **caches** the result for the rest of this run:
    an identical call later in the same run (same stage, technique, params and inputs)
    returns the cached output without re-running the technique — it is traced again
    with ``cached: True``, and its cost, tokens and energy are counted once.

Calls are compared by value. If every param and input is plain JSON (str, int, float,
bool, None, and lists and str-keyed dicts of them), a call is keyed by its JSON; otherwise
by its pickle — type and full content, never ``str()`` — so two different numpy arrays
that print alike never share an entry. A call with a value that can't be pickled (a
lambda, an open client) is not cached: it runs every time.

``ctx.run(stage, ..., cache=False)`` opts one call out, e.g. to sample a stochastic stage
several times (best-of-n, self-consistency): it always runs, is traced with
``cached: False`` and pays its own cost, tokens and energy, and it neither reads nor
fills the cache — a later cached call returns what it would have returned without it.

The cache lives on the ``Context``, and a ``Pipeline`` creates a fresh one for every
(config, item, replication) cell, so nothing is shared between cells: each configuration
executes, and pays for, every stage it reaches — even a stage another configuration ran
with identical inputs.

The control flow around these calls is just your Python — branches, cascades,
loops — so the topology is entirely yours; CAFE only watches the calls.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import pickle
import time
from typing import TYPE_CHECKING, Any

from cafe import llm

if TYPE_CHECKING:
    from cafe.techniques.registry import TechniqueSpec

# The step currently executing in this task, as (owning Context, costs added via
# ``add_cost``). A ContextVar, like ``llm._usage_sink``: each ``asyncio`` task gets its own
# copy, so steps run concurrently (``asyncio.gather``) or nested (``ctx.run`` inside a
# technique) each accrue their own manual cost instead of sharing one accumulator.
_manual_cost_sink: contextvars.ContextVar[tuple[Context, list[float]] | None] = (
    contextvars.ContextVar("cafe_manual_cost_sink", default=None)
)


# Names a technique's runtime input can't have: ``ctx.run`` takes ``stage`` and ``cache``
# itself, so an input with either name could never be passed (``Pipeline`` refuses it).
RESERVED_INPUT_NAMES = frozenset({"stage", "cache"})

# Exact types only: json.dumps writes a subclass (an IntEnum, a numpy float64) as its base
# type, so it would share a key with a plain value of that type.
_JSON_SCALARS = (str, int, float, bool, type(None))


def _plain_json(value: Any) -> bool:
    """Whether ``json.dumps`` writes ``value`` without changing what it is.

    True for plain str/int/float/bool/None and for lists and str-keyed dicts of those; false
    for anything json would convert (a tuple becomes a list, an int key a str) or that only
    ``default=str`` could write.
    """
    kind = type(value)
    if kind in _JSON_SCALARS:
        return True
    if kind is list:
        return all(_plain_json(v) for v in value)
    if kind is dict:
        return all(type(k) is str and _plain_json(v) for k, v in value.items())
    return False


def _cache_key(stage: str, name: str, params: dict, inputs: dict) -> str | None:
    """The stage-cache key of one call, or ``None`` when the call can't be cached safely.

    Plain JSON payloads are keyed by their canonical JSON. Anything else is keyed by the
    SHA-256 of its pickle (type and full content), never by ``str()``: a long numpy array
    prints with "...", so two different arrays can print alike and would share an entry. A
    payload that can't be pickled gets no key, so the call runs instead of risking a hit
    on another input's output.
    """
    payload = {"stage": stage, "name": name, "params": params, "inputs": inputs}
    try:
        if _plain_json(payload):
            return json.dumps(payload, sort_keys=True)
        blob = pickle.dumps(
            (stage, name, sorted(params.items()), sorted(inputs.items())),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    except Exception:  # can't be pickled (a lambda, a lock, a client), or nested too deep
        return None
    return "pickle:" + hashlib.sha256(blob).hexdigest()


class Context:
    """Runs your techniques and records what happened. One per (config, item) run."""

    def __init__(self, techniques: dict[tuple[str, str], "TechniqueSpec"],
                 config: dict[str, Any]) -> None:
        self._techniques = techniques   # this pipeline's registry — NOT a global
        self.config = config
        self.trace: list[dict[str, Any]] = []
        self.total_cost: float = 0.0
        self.total_tokens: int = 0
        self.total_energy_wh: float = 0.0
        self.energy_declared: bool = False  # True once any executed technique declared energy
        self._cache: dict[str, Any] = {}

    def add_cost(self, usd: float) -> None:
        """Charge an extra cost to the current stage — for a variable, non-LLM price
        the technique computes itself (e.g. ``ctx.add_cost(0.001 * len(hits))``).

        Called inside a technique, the cost is attributed to that technique's step, also
        when several steps run at once (``asyncio.gather``) or one ``ctx.run`` is nested
        in another. Called from the compose function itself (outside any ``ctx.run``),
        it is added to the run's total ``cost_usd`` without belonging to any stage.

        Parameters
        ----------
        usd:
            The amount in US dollars.
        """
        current = _manual_cost_sink.get()
        if current is not None and current[0] is self:
            current[1].append(float(usd))
        else:
            self.total_cost += float(usd)

    async def run(self, stage: str, *, cache: bool = True, **inputs: Any) -> Any:
        """Run whichever technique the config selected for ``stage`` and return its output.

        The step is timed, metered and appended to :attr:`trace`. A call identical to an
        earlier cached one in this run (same stage, technique, params and inputs) returns
        that call's output without running the technique: it is traced with
        ``cached: True`` and adds no cost, tokens or energy. Params and inputs are compared
        by value — as JSON, or else by their pickle, never by ``str()`` — and a call with a
        value that can't be pickled is never cached.

        Parameters
        ----------
        stage:
            The stage to run. The config picks its technique; a stage with a single
            registered technique needs no factor.
        cache:
            ``False`` runs the technique even if an identical call already ran — to sample
            a stochastic stage several times (best-of-n, self-consistency). Each such call
            is traced with ``cached: False`` and pays its own cost, tokens and energy; it
            neither reads nor fills the cache, so a later cached call returns what it
            would have returned without it. Keyword-only and never passed on to the
            technique, so no technique can take an input named ``cache``.
        **inputs:
            The technique's runtime inputs (its arguments without defaults), by name. Its
            tunable params come from the config, not from here.

        Returns
        -------
        Any
            The technique's output (the ``"output"`` value if it returned an
            ``{"output": ..., "cost_usd": ...}`` dict). A stage skipped by a ``None`` level
            returns its single input unchanged.

        Raises
        ------
        TypeError
            If ``cache`` is not a bool — e.g. a value meant for a technique input named
            ``cache``, which ``ctx.run`` keeps for itself.
        KeyError
            If no technique is registered for ``stage``, several are but no factor picks
            one, the config picks one this pipeline doesn't have, or a stage skipped by a
            ``None`` level gets more than one input.
        """
        if not isinstance(cache, bool):
            raise TypeError(
                f"ctx.run({stage!r}, cache=...) takes True or False, got "
                f"{type(cache).__name__}; 'cache' is ctx.run's own switch for the stage cache, "
                "so a technique can't take an input named 'cache' — rename that input"
            )
        if stage in self.config:
            name = self.config[stage]
        else:
            # No factor picked a technique for this stage. If exactly ONE technique is
            # registered for it, it's a FIXED stage — just use it (no dummy 1-level factor
            # needed). Several techniques with no factor is ambiguous → a clear error.
            here = [n for (s, n) in self._techniques if s == stage]
            if len(here) == 1:
                name = here[0]
            elif not here:
                raise KeyError(
                    f"stage {stage!r} has no registered technique and no factor selects one"
                )
            else:
                raise KeyError(
                    f"stage {stage!r} has {len(here)} techniques {here} but no factor to pick "
                    f"one — add a factor named {stage!r} to the study"
                )

        # A ``None`` level — cafe.Factor("rerank", ["by_length", None]) — means "skip this
        # stage": pass the sole input straight through. (A technique literally named "none"
        # is not special — it just runs. Multi-input stages: pipe.factor(none="<input>").)
        if name is None:
            if len(inputs) != 1:
                raise KeyError(
                    f"stage {stage!r} is skipped (None level) but got {len(inputs)} inputs "
                    f"{sorted(inputs)}; use pipe.factor({stage!r}, "
                    "none='<input to pass through>') to say which one to keep"
                )
            self.trace.append({
                "stage": stage, "technique": "none", "params": {},
                "elapsed_s": 0.0, "cost_usd": 0.0, "tokens": 0, "cached": False,
            })
            return next(iter(inputs.values()))

        try:
            spec = self._techniques[(stage, str(name))]
        except KeyError:
            avail = [n for (s, n) in self._techniques if s == stage]
            raise KeyError(
                f"no technique {name!r} registered for stage {stage!r} on this pipeline; "
                f"have: {avail}"
            ) from None

        params = {
            pname: self.config.get(f"{stage}.{pname}", default)
            for pname, default in spec.params.items()
        }

        # No key (cache=False, or inputs CAFE can't fingerprint exactly): run, and don't store.
        key = _cache_key(stage, str(name), params, inputs) if cache else None
        if key is not None and key in self._cache:
            hit = self._cache[key]
            self.trace.append({**hit["meta"], "cached": True})
            return hit["output"]

        # Collect any LLM usage made *inside* this technique so we attribute
        # tokens + cost to this stage automatically (the user just calls complete).
        sink: list[dict[str, Any]] = []
        manual: list[float] = []
        token = llm._usage_sink.set(sink)
        manual_token = _manual_cost_sink.set((self, manual))
        t0 = time.monotonic()
        try:
            raw = await spec.fn(self, **inputs, **params)
        finally:
            _manual_cost_sink.reset(manual_token)
            llm._usage_sink.reset(token)
        elapsed = round(time.monotonic() - t0, 5)

        auto_tokens = sum(u["tokens"] for u in sink)
        auto_cost = sum(u["cost_usd"] for u in sink)

        # A technique may also return a plain value, or a dict {"output":…, "cost_usd":…}
        # to declare a cost explicitly (e.g. a priced API CAFE can't see).
        output, declared_cost = raw, 0.0
        if isinstance(raw, dict) and "output" in raw:
            output = raw["output"]
            declared_cost = float(raw.get("cost_usd", 0.0) or 0.0)
        # Total = LLM cost + the technique's fixed cost_usd + anything via ctx.add_cost()
        # + an explicit cost in the return dict.
        cost = round(auto_cost + spec.cost_usd + sum(manual) + declared_cost, 6)

        meta = {
            "stage": stage,
            "technique": str(name),
            "params": params,
            "elapsed_s": elapsed,
            "cost_usd": cost,
            "tokens": auto_tokens,
            "cached": False,
        }
        # Energy is opt-in per technique: computed (and traced) only when declared.
        if spec.energy_wh is not None or spec.energy_wh_per_1k_tokens is not None:
            energy = (spec.energy_wh or 0.0) + (spec.energy_wh_per_1k_tokens or 0.0) * auto_tokens / 1000
            meta["energy_wh"] = round(energy, 6)
            self.total_energy_wh += energy
            self.energy_declared = True
        self.total_cost += cost
        self.total_tokens += auto_tokens
        self.trace.append(meta)
        if key is not None:
            self._cache[key] = {"output": output, "meta": meta}
        return output
