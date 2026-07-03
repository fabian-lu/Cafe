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
    design_options: dict[str, Any] = field(default_factory=dict)
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

    # ── What this study will do, before running it ─────────────────────────────
    def plan(self) -> dict[str, Any]:
        """The run plan as numbers: configs, items, reps, total runs (and judge calls).

        Cheap — it expands the design but runs nothing. Useful for a sanity check or a
        cost estimate before spending tokens.
        """
        from cafe.design import size

        n_configs = size(self)
        n_items = max(1, len(self.dataset))
        runs = n_configs * n_items * self.replications
        judged = self.judge is not None and self.rubric is not None
        plan: dict[str, Any] = {
            "name": self.name,
            "design": self.design,
            "factors": [f.name for f in self.factors],
            "configs": n_configs,
            "items": n_items,
            "replications": self.replications,
            "runs": runs,
            "judged": judged,
        }
        if judged:
            plan["judge_calls"] = runs * self.judge_replications
        return plan

    def check(self) -> list[str]:
        """Advisory warnings about whether the design has enough data for the stats —
        cheap, runs nothing, meant to be read **before** spending tokens. Flags thin
        designs that make the mixed-effects models (linear / CLMM / logistic) unstable.
        Returns a list of human-readable strings (empty = nothing obviously wrong).

        Not a power analysis (mixed-model power needs simulation) — just heuristics on the
        numbers. Surfaced by :meth:`preflight` and emitted once when :func:`cafe.evaluate`
        starts.
        """
        from cafe.design import size

        warns: list[str] = []
        n_items = len(self.dataset)
        try:
            n_configs = size(self)
        except Exception:  # noqa: BLE001
            n_configs = 0
        scale = getattr(getattr(self.rubric, "scale_type", None), "value", None)

        # Enough questions to estimate the per-question random intercept?
        if n_items == 0:
            warns.append("no dataset items — nothing to run")
        elif n_items < 2:
            warns.append(
                f"only {n_items} input — the mixed-effects models need ≥2 questions for the "
                "per-question random intercept; they'll drop to a fixed-effects fit"
            )
        elif n_items < 8:
            warns.append(
                f"{n_items} inputs — the mixed-effects models (linear / CLMM / logistic) estimate a "
                "per-question random intercept; with fewer than ~8 questions those estimates can be "
                "near-singular/unstable. Add inputs for reliable p-values"
            )

        # Thin design: many configurations relative to the number of questions.
        if n_items and n_configs > n_items:
            warns.append(
                f"{n_configs} configurations but only {n_items} inputs — few observations per "
                "configuration; per-factor estimates will be weak. Add inputs, or reduce factors/levels"
            )

        # Binary rubric with few items → a factor may perfectly separate pass/fail.
        if scale == "binary" and 0 < n_items < 10:
            warns.append(
                f"binary rubric with {n_items} inputs — a factor may perfectly separate pass/fail, "
                "making the logistic model degenerate (odds ratios diverge). More inputs reduce this risk"
            )

        # Design shape: can this study actually compare anything?
        multi_level = [f for f in self.factors if len(f.levels) >= 2]
        if len(multi_level) == 0:
            warns.append(
                "no factor varies across ≥2 levels — there is nothing to compare; add a factor "
                "with at least two levels (e.g. an on/off feature to test 'does X help?')"
            )
        elif len(multi_level) == 1:
            warns.append(
                "only one factor varies — you can estimate its main effect but no interactions "
                "(you can't ask whether two techniques help *together*); add a second factor for that"
            )

        # Judge / rubric must both be set to judge; one without the other silently skips judging.
        if (self.judge is None) != (self.rubric is None):
            has, missing = ("judge", "rubric") if self.rubric is None else ("rubric", "judge")
            warns.append(
                f"a {has} is set but no {missing} — judging is skipped unless BOTH are provided "
                "(the study will only generate answers)"
            )

        # Judge replications at temperature 0 measure nothing (a deterministic judge won't vary).
        temp = getattr(self.judge, "temperature", None)
        if self.judge_replications >= 2 and temp == 0.0:
            warns.append(
                f"judge_replications={self.judge_replications} but the judge temperature is 0.0 — "
                "a deterministic judge won't vary between repetitions, so this measures no judge "
                "noise; raise the judge temperature to measure it"
            )

        # FactorType.continuous is advisory only — the stats layer treats every factor as categorical.
        continuous = [f.name for f in self.factors if getattr(f.type, "value", None) == "continuous"]
        if continuous:
            warns.append(
                f"factor(s) {continuous} are declared continuous, but the statistics currently treat "
                "every factor as categorical (each level is a separate group; no trend/slope is fit). "
                "Numeric-covariate modeling is planned — for now FactorType is advisory"
            )
        return warns

    def __repr__(self) -> str:
        try:
            p = self.plan()
        except Exception:  # noqa: BLE001 — repr must never raise
            return f"Study(name={self.name!r}, factors={[f.name for f in self.factors]})"
        head = (
            f"Study({p['name']!r}: {p['configs']} configs × {p['items']} items "
            f"× {p['replications']} reps = {p['runs']} runs"
        )
        parts = []
        if p["design"] != "full_factorial":
            parts.append(p["design"])
        parts.append("factors: " + (", ".join(p["factors"]) or "—"))
        if p["judged"]:
            jr = f" ×{self.judge_replications}" if self.judge_replications > 1 else ""
            parts.append(f"judged by {getattr(self.judge, 'model', 'judge')}{jr}")
        return head + "; " + "; ".join(parts) + ")"

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
        from cafe.evaluation import emit_design_warnings, evaluate

        # Emit design-adequacy warnings here — synchronously, before the event loop — so
        # they point at the caller's code rather than at asyncio internals.
        emit_design_warnings(self)
        return self._run_blocking(lambda: evaluate(self, _warn_design=False, **kwargs))

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

        Shows the messages exactly as sent, with role labels (``[SYSTEM]`` / ``[USER]``)
        so the system framing and the rubric-derived user prompt are both visible. Uses
        the study's rubric + judge and, by default, the first dataset item. Print it to
        verify the judging before spending any tokens.
        """
        if self.judge is None or self.rubric is None:
            raise ValueError("study has no judge/rubric set to preview")
        if item is None:
            item = self.dataset[0] if self.dataset else ""
        question = item["text"] if isinstance(item, dict) and "text" in item else str(item)
        reference = item.get("reference") if isinstance(item, dict) else None
        if not hasattr(self.judge, "preview"):
            raise ValueError("this judge has no .preview() to show a prompt (use an LLMJudge)")
        return self.judge.preview(self.rubric, question, answer, reference)

    def _run_blocking(self, make_coro):
        from cafe._async import run_blocking

        return run_blocking(make_coro)
