"""Study, Factor — the user-facing description of an experiment.

A Study is the black box (the system under test) plus the factors to vary and
the dataset to run it on. It carries no execution or persistence logic itself;
:func:`cafe.run_study` consumes it.

CAFE does not model your pipeline's topology. A factor is just a named axis with
levels; your system reads the chosen levels from the ``config`` dict and does
whatever it wants internally — RAG, routing, a cascade, an agent. That black-box
stance is what makes CAFE general across compound AI systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FactorType(str, Enum):
    """How a factor is treated by design generation and (later) statistics."""

    categorical = "categorical"  # unordered: reranker in {none, cross_encoder}
    ordinal = "ordinal"          # ordered: effort in {low, med, high}
    continuous = "continuous"    # numeric knob: temperature, top_k


@dataclass
class Factor:
    """One axis of the experiment.

    ``levels`` is the set of values this factor can take. Full factorial visits
    every combination of every factor's levels — there is no limit on the number
    of factors or levels (that limit only applies to fractional/screening designs).
    """

    name: str
    levels: list[Any]
    type: FactorType = FactorType.categorical

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("factor name must be a non-empty string")
        self.levels = list(self.levels)
        if len(self.levels) == 0:
            raise ValueError(f"factor {self.name!r} must have at least one level")
        if isinstance(self.type, str):
            self.type = FactorType(self.type)


@dataclass
class Study:
    """A complete evaluation: a system, the factors to vary, the inputs, and how
    to judge the results.

    Fields
    ------
    system:
        The black box under test, runnable as ``run(config, item) -> output`` —
        a plain callable (sync or async) or an object with such a method.
    factors:
        The axes to vary. Their Cartesian product (full factorial) defines the
        configurations.
    dataset:
        The evaluation set, one element per item. An item may be any value; if it
        is a mapping it may carry ``"id"`` (for resume), ``"text"`` (the question
        shown to the judge), and ``"reference"`` (a gold answer for the judge).
    rubric:
        The :class:`cafe.Rubric` the judge scores answers on. Optional —
        leave ``None`` to only generate answers without judging.
    judge:
        The :class:`cafe.LLMJudge` (or any object with a compatible ``score``
        method). Optional, paired with ``rubric``.
    replications:
        How many times each (configuration, input) is executed. This is how CAFE
        measures the *system's* run-to-run nondeterminism.
    judge_replications:
        How many times the judge re-scores each answer — measures the *judge's*
        own nondeterminism, separately from the system's.
    """

    name: str
    system: Any
    factors: list[Factor] = field(default_factory=list)
    dataset: list[Any] = field(default_factory=list)
    rubric: Any = None
    judge: Any = None
    design: str = "full_factorial"
    replications: int = 1
    judge_replications: int = 1

    def __post_init__(self) -> None:
        names = [f.name for f in self.factors]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate factor names: {names}")
        if self.replications < 1:
            raise ValueError("replications must be >= 1")
        if self.judge_replications < 1:
            raise ValueError("judge_replications must be >= 1")

    # ── Running, synchronously, from anywhere ──────────────────────────────────
    #
    # These wrappers work in plain scripts AND inside an already-running event
    # loop (e.g. a Jupyter notebook): if a loop is running we execute in a worker
    # thread so we never hit "asyncio.run() cannot be called from a running event
    # loop". Async-native callers can instead await the module functions
    # (``cafe.evaluate``, ``cafe.run_study``, ``cafe.preflight``).

    def evaluate(self, **kwargs: Any):
        """Run the **complete** evaluation: generate answers, judge them (if a
        rubric + judge are set), and attribute quality to the factors. Returns an
        :class:`cafe.evaluation.Evaluation`."""
        from cafe.evaluation import evaluate

        return self._run_blocking(lambda: evaluate(self, **kwargs))

    def preflight(self, **kwargs: Any):
        """Quick check before a full run: one input through every configuration,
        no replication or judging, plus a cost/time estimate. Returns a
        :class:`cafe.evaluation.Preflight`."""
        from cafe.evaluation import preflight

        return self._run_blocking(lambda: preflight(self, **kwargs))

    def run(self, **kwargs: Any):
        """Lower-level: generate answers only (no judging). Returns
        :class:`cafe.Results`. Most users want :meth:`evaluate`."""
        from cafe.execution import run_study

        return self._run_blocking(lambda: run_study(self, **kwargs))

    def preview_judge_prompt(self, answer: str, item: Any = None) -> str:
        """Return the exact judge prompt for an example ``answer`` — no LLM call.

        Uses the study's rubric + judge and, by default, the first dataset item.
        Print it to verify the judging is doing what you intend before spending
        any tokens.
        """
        if self.judge is None or self.rubric is None:
            raise ValueError("study has no judge/rubric set to preview")
        if item is None:
            item = self.dataset[0] if self.dataset else ""
        question = item["text"] if isinstance(item, dict) and "text" in item else str(item)
        reference = item.get("reference") if isinstance(item, dict) else None
        return self.judge.render_prompt(self.rubric, question, answer, reference)

    def _run_blocking(self, make_coro):
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(make_coro())

        import threading

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
