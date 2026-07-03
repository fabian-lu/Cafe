"""High-level orchestration: evaluate a study end to end, or preflight it.

``evaluate`` is the one call most users want — it generates answers, judges them
(when the study has a rubric + judge), and attributes quality to the factors,
returning a single :class:`Evaluation`. The lower-level phases
(:func:`cafe.run_study`, :func:`cafe.judge_results`, :func:`cafe.attribute`)
remain available for advanced/phased use.

``preflight`` is the deliberately-cheap check you run *before* a full study: one
input through every configuration, no replication, no judging, plus a cost/time
estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from cafe.design import size
from cafe.execution import _input_id, estimate, run_study
from cafe.judging import Ratings, judge_results
from cafe.execution.results import Observation, Results
from cafe.stats import Attribution, attribute

if TYPE_CHECKING:
    from cafe.study import Study

ProgressFn = Callable[[Observation, int, int], None]


@dataclass
class Evaluation:
    """The complete result of evaluating a study: answers, ratings, attribution.

    Everything is reachable as plain data: ``answers.observations`` (each
    ``Observation``), ``ratings.items`` (each ``Rating``, with ``prompt`` /
    ``raw_response`` / ``value_numeric``), ``attribution`` / ``effects`` / ``clmm``,
    and the ``answers.df`` / ``ratings.df`` DataFrames. :meth:`records` joins it all
    into one row-per-verdict view.
    """

    study_name: str
    answers: Results
    ratings: Ratings | None = None
    attribution: Attribution | None = None
    questions: dict[str, str] = field(default_factory=dict)    # input_id -> question text
    references: dict[str, str] = field(default_factory=dict)   # input_id -> gold answer

    _effects_cache: Any = field(default=None, repr=False, compare=False)

    _clmm_cache: Any = field(default=None, repr=False, compare=False)

    _logistic_cache: Any = field(default=None, repr=False, compare=False)

    def records(self) -> list[dict[str, Any]]:
        """One row per judge verdict, joining **everything** for inspection/export:
        question, reference, the factor levels, the answer, per-answer cost/latency, the
        full judge prompt (system + user), the raw judge response, and the parsed verdict.

        This is the "give me everything" view — ``import pandas as pd;
        pd.DataFrame(ev.records())`` and slice however you like. With no judge it falls
        back to one row per answer.
        """
        by_key = {o.key(): o for o in self.answers.observations}
        sys_prompt = self.ratings.judge_system_prompt if self.ratings else None
        rows: list[dict[str, Any]] = []

        if self.ratings is None or not self.ratings.items:
            for o in self.answers.observations:
                meta = o.metadata or {}
                rows.append({
                    "input_id": o.input_id,
                    "question": self.questions.get(o.input_id),
                    "reference": self.references.get(o.input_id),
                    **o.config, "rep": o.rep, "answer": o.output,
                    "elapsed_s": o.elapsed_s,
                    "cost_usd": meta.get("cost_usd"), "tokens": meta.get("tokens"),
                    "error": o.error,
                })
            return rows

        for r in self.ratings.items:
            obs = by_key.get(r.obs_key)
            meta = (obs.metadata if obs else {}) or {}
            rows.append({
                "input_id": r.input_id,
                "question": self.questions.get(r.input_id),
                "reference": self.references.get(r.input_id),
                **r.config,
                "rep": r.rep,
                "judge_rep": r.judge_rep,
                "answer": obs.output if obs else None,
                "elapsed_s": obs.elapsed_s if obs else None,
                "cost_usd": meta.get("cost_usd"),
                "tokens": meta.get("tokens"),
                "judge_system": sys_prompt,
                "judge_prompt": r.prompt,
                "judge_raw": r.raw_response,
                "verdict": r.value_numeric,
                "reasoning": r.reasoning,
                "error": r.error,
            })
        return rows

    @property
    def overall_mean(self) -> float | None:
        """Mean verdict across all usable answers (judge replications averaged), or
        ``None`` if unjudged. The single-number summary — e.g. a pass rate for a binary
        rubric. Per-configuration / per-factor breakdowns live on ``attribution``."""
        return self.attribution.overall_mean if self.attribution is not None else None

    @property
    def marginal_means(self):
        """Per-factor marginal means as a tidy DataFrame (factor, level, mean, n) — the
        "which level scores higher" table, ready to drop into a paper. ``None`` if
        unjudged. (The same data is in ``attribution.factor_marginals``.)"""
        if self.attribution is None:
            return None
        import pandas as pd

        return pd.DataFrame(self.attribution.factor_marginals)

    @property
    def residuals(self):
        """Per-answer residuals from the inferential fit (like R's ``residuals(fit)``),
        or ``None`` if unjudged. Handy for diagnostics (normality, outliers)."""
        eff = self.effects
        return list(eff.residuals) if eff is not None and eff.residuals else None

    @property
    def fitted(self):
        """Per-answer fitted values from the inferential fit (like R's ``fitted(fit)``),
        or ``None`` if unjudged."""
        eff = self.effects
        return list(eff.fitted) if eff is not None and eff.fitted else None

    @property
    def variance_components(self):
        """The mixed model's ``{random_intercept, residual}`` variances (like R's
        ``VarCorr(fit)``), or ``None`` if unjudged / only a fixed-effects fit was possible."""
        eff = self.effects
        return eff.variance_components if eff is not None else None

    @property
    def effects(self):
        """Inferential statistics (mixed model → F/p, partial η², Cohen's d).

        Computed lazily on first access from the ratings; cached thereafter.
        Returns ``None`` if the study wasn't judged.
        """
        if self.ratings is None:
            return None
        if self._effects_cache is None:
            from cafe.stats import fit_effects

            self._effects_cache = fit_effects(self.ratings)
        return self._effects_cache

    @property
    def clmm(self):
        """Ordinal CLMM (R) for ordinal rubrics. Lazy; returns ``None`` if unjudged.

        The result's ``available`` flag says whether R produced a fit; if not, its
        ``reason`` explains why (e.g. R not installed) — see ``cafe doctor``.
        """
        if self.ratings is None:
            return None
        if self._clmm_cache is None:
            from cafe.stats import fit_clmm

            self._clmm_cache = fit_clmm(self.ratings)
        return self._clmm_cache

    @property
    def logistic(self):
        """Binary logistic model (log-odds / odds ratios) for **binary** pass/fail
        rubrics. Lazy; returns ``None`` if unjudged. ``available=False`` (with a
        ``reason``) for non-binary rubrics or degenerate data."""
        if self.ratings is None:
            return None
        if self._logistic_cache is None:
            from cafe.stats import fit_logistic

            self._logistic_cache = fit_logistic(self.ratings)
        return self._logistic_cache

    def report(self, *, interactions: int = 2) -> str:
        """The **full** statistical picture in one string: a pipeline summary, the
        descriptive layer, then the **model matched to the rubric's scale type** —
        numeric → linear mixed model, ordinal → linear + cumulative-link mixed model,
        binary → logistic. So each scale is analysed with its statistically correct model.

        ``interactions`` is the max interaction order to model (2 = also two-way, the
        default; 1 = main effects only). The cheap ``repr`` shows only the descriptive
        layer (so displaying a result is instant); ``report()`` additionally fits the
        models, so the first call may take a moment. Print it::

            print(result.report())
        """
        from cafe.judging.rubric import ScaleType

        bar = "─" * 60
        parts = [self.show(), "", self._pipeline_line()]
        warns = self._health_warnings()
        if warns:
            parts.append("")
            parts += [f"⚠ {w}" for w in warns]
        if self.attribution is not None:
            parts += ["", bar, "DESCRIPTIVE — means & best configuration", bar,
                      self.attribution.show()]
        if self.ratings is not None:
            from cafe.stats import fit_clmm, fit_effects, fit_logistic

            scale = getattr(getattr(self.ratings, "rubric", None), "scale_type", None)
            if scale == ScaleType.binary:
                log = self.logistic if interactions == 2 else fit_logistic(self.ratings, interactions=interactions)
                parts += ["", bar, "LOGISTIC — binary pass/fail model (log-odds & odds ratios)",
                          bar, log.show()]
            elif scale == ScaleType.numeric:
                eff = self.effects if interactions == 2 else fit_effects(self.ratings, interactions=interactions)
                parts += ["", bar, "LINEAR — Gaussian mixed-effects model (significance & effect sizes)",
                          bar, eff.show()]
            else:  # ordinal (and the default): the linear view + the correct ordinal CLMM
                eff = self.effects if interactions == 2 else fit_effects(self.ratings, interactions=interactions)
                parts += ["", bar, "INFERENTIAL — mixed-effects model (significance & effect sizes)",
                          bar, eff.show()]
                clmm = self.clmm if interactions == 2 else fit_clmm(self.ratings, interactions=interactions)
                parts += ["", bar, "ORDINAL — cumulative-link mixed model (verdicts as ordered categories)",
                          bar, clmm.show()]
        return "\n".join(parts)

    def rejudge(self, judge: Any, *, rubric: Any = None, repetitions: int = 1,
                name: str | None = None, concurrency: int = 8,
                checkpoint_path: str | None = None, progress: bool = True):
        """Score the **same answers** again — with a different judge, ``rubric``, or
        number of ``repetitions`` — returning a new ``Evaluation`` (reusing this one's
        questions/references; nothing is regenerated). This is the "generate once, judge
        many ways" path:

            free = result.rejudge(cafe.LLMJudge(model=m, preset="single_answer"))
            biny = result.rejudge(judge, rubric=cafe.rubrics.CORRECT_PASS_FAIL)
            noise = result.rejudge(judge, repetitions=3)   # the judge's own spread

        ``checkpoint_path`` makes the judging crash-safe/resumable (verdicts appended as
        they land). Also handy for judge↔judge reliability —
        ``cafe.reliability(raters={"a": result, "b": result.rejudge(judge_b)})``.
        """
        from cafe._async import run_blocking
        from cafe.judging import judge_results
        from cafe.stats import attribute

        rubric = rubric or getattr(self.ratings, "rubric", None)
        if rubric is None:
            raise ValueError("rejudge needs a rubric — this evaluation wasn't judged")
        ratings = run_blocking(lambda: judge_results(
            self.answers, judge, rubric, repetitions=repetitions, concurrency=concurrency,
            questions=self.questions, references=self.references,
            checkpoint_path=checkpoint_path, progress=progress,
        ))
        return Evaluation(
            study_name=name or f"{self.study_name} ({getattr(judge, 'model', 'judge')})",
            answers=self.answers, ratings=ratings, attribution=attribute(ratings),
            questions=self.questions, references=self.references,
        )

    def judge_stability(self):
        """How consistently the judge scored the **same answer** across its repetitions —
        the judge analogue of system ``replications``. Returns a
        :class:`cafe.stats.JudgeStability` (per-answer std dev + a summary). Only
        informative when the judge scored answers more than once (``judge_replications``
        on the study, or ``rejudge(..., repetitions=k)``)::

            noisy = result.rejudge(judge, repetitions=3)
            print(noisy.judge_stability().show())
        """
        from cafe.stats import judge_stability as _judge_stability

        if self.ratings is None:
            raise ValueError("no ratings to measure — this evaluation wasn't judged")
        return _judge_stability(self.ratings)

    def pipeline(self):
        """The composed pipeline (stage order + levels), observed from the run's traces.
        Only meaningful for a composed system; returns an empty pipeline otherwise."""
        from cafe.techniques.composed import pipeline as _pipeline

        return _pipeline(self)

    def plot(self, kind: str | None = None, **kwargs):
        """Plots for this evaluation. No argument → a **dashboard** of the key plots; or
        pass a ``kind``: ``"marginals"``, ``"interaction"``, ``"configs"``,
        ``"distribution"``, ``"effects"``, ``"pareto"``. Returns the matplotlib Figure
        (dashboard) or Axes (single plot) — ``.savefig(...)`` or tweak it freely::

            result.plot()                 # everything
            result.plot("interaction")    # just one
        """
        from cafe.stats.plots import plot as _plot

        return _plot(self, kind, **kwargs)

    def _pipeline_line(self) -> str:
        """A funnel: answers attempted → judged → usable, so dropped cells are visible."""
        n_ans, n_failed = len(self.answers), len(self.answers.errors)
        seg = f"pipeline: {n_ans} answers"
        if n_failed:
            seg += f" ({n_failed} failed to generate)"
        if self.ratings is not None:
            n_bad = len(self.ratings.errors)
            seg += f"  →  {len(self.ratings)} judged"
            if n_bad:
                seg += f" ({n_bad} unparseable)"
            seg += f"  →  {len(self.ratings) - n_bad} usable verdict(s)"
        return seg

    def _health_warnings(self) -> list[str]:
        """Problems worth flagging: failed generations (which unbalance the design) and
        unparseable verdicts — distinguished, since they happen at different stages."""
        from collections import Counter

        from cafe.execution.results import config_label

        out: list[str] = []
        failed = self.answers.errors
        if failed:
            by = Counter(config_label(o.config) for o in failed)
            top = ", ".join(f"{c} ({n})" for c, n in by.most_common(2))
            out.append(
                f"{len(failed)} of {len(self.answers)} answers failed to generate "
                f"(mostly {top}) — the design is unbalanced; inspect result.answers.errors"
            )
        if self.ratings is not None and self.ratings.errors:
            errs = self.ratings.errors
            unjudgeable = [r for r in errs if (r.error or "").startswith("unjudgeable")]
            unparseable = [r for r in errs if r not in unjudgeable]
            if unjudgeable:
                out.append(
                    f"{len(unjudgeable)} answer(s) produced no output (None) and could not be "
                    "judged — they are recorded as errors; inspect result.ratings.failures()"
                )
            if unparseable:
                out.append(
                    f"{len(unparseable)} verdict(s) were unparseable — "
                    "inspect result.ratings.failures()"
                )
        return out

    def summary(self) -> dict[str, Any]:
        out = self.answers.summary()
        if self.ratings is not None:
            out["n_ratings"] = len(self.ratings)
            out["n_rating_errors"] = len(self.ratings.errors)
        if self.attribution is not None and self.attribution.best_config is not None:
            out["best_config"] = self.attribution.best_config["config"]
        return out

    def show(self) -> str:
        s = self.summary()
        line = f"{s['n_answers']} answers"
        if s.get("n_errors"):
            line += f" ({s['n_errors']} failed)"
        line += f" · {s['n_configs']} configs · {s['n_inputs']} inputs"
        if "n_ratings" in s:
            line += f" · {s['n_ratings']} ratings"
        best = s.get("best_config")
        if best:
            line += " · best: " + "·".join(f"{k}={best[k]}" for k in sorted(best))
        return line

    def __repr__(self) -> str:
        return f"Evaluation({self.show()})"

    def _repr_html_(self) -> str:  # pragma: no cover - notebook display
        parts = [f"<b>Evaluation</b> — {self.show()}"]
        if self.attribution is not None:
            parts.append(f"<pre>{self.attribution.show()}</pre>")
        return "".join(parts)


@dataclass
class Preflight:
    """A quick pre-run check: one input per configuration, plus an estimate."""

    answers: Results
    estimate: dict[str, Any]
    warnings: list[str] = field(default_factory=list)   # design-adequacy advisories
    judge_calls: int = 0                                 # judge calls the full study will make

    def show(self) -> str:
        lines = ["preflight — one input through every configuration:"]
        for o in self.answers.observations:
            cfg = "·".join(f"{k}={o.config[k]}" for k in sorted(o.config))
            body = (o.output or o.error or "").replace("\n", " ")
            lines.append(f"  [{cfg}] {body[:80]}")
        e = self.estimate
        compute = e.get("est_total_compute_s")
        cost = e.get("est_total_cost_usd")
        compute_s = f"~{compute}s" if compute is not None else "n/a"
        # Cost is only available if the system returns a `cost_usd` per answer.
        cost_s = f"~${cost}" if cost is not None else "n/a (system reports no cost_usd)"
        lines.append("")
        lines.append(f"full study: {e['total_cells']} cells; est. compute {compute_s}, est. cost {cost_s}")
        # The estimate is answer generation only — the judge phase is not measured here.
        if self.judge_calls:
            lines.append(
                f"  note: this estimate covers answer generation only; the study will also make "
                f"~{self.judge_calls} judge call(s) (time/cost not included)."
            )
        if self.warnings:
            lines.append("")
            lines.append("design check — read before spending tokens:")
            lines += [f"  ⚠ {w}" for w in self.warnings]
        return "\n".join(lines)


def _question_and_reference_maps(study: "Study") -> tuple[dict[str, str], dict[str, str]]:
    """Derive ``input_id -> question`` and ``input_id -> reference`` from the
    study's inputs, so the judge sees the original question and any gold answer
    without the caller wiring it up by hand."""
    questions: dict[str, str] = {}
    references: dict[str, str] = {}
    for idx, item in enumerate(study.dataset):
        iid = _input_id(item, idx)
        if isinstance(item, dict):
            questions[iid] = str(item.get("text", item.get("question", "")))
            if item.get("reference") is not None:
                references[iid] = str(item["reference"])
        else:
            questions[iid] = str(item)
    return questions, references


def emit_design_warnings(study: "Study") -> None:
    """Emit the study's design-adequacy advisories via ``warnings.warn``. The sync
    entry points (``Study.evaluate`` etc.) call this *before* entering the event loop so
    the warning points at the user's code, not at asyncio internals."""
    import warnings as _warnings

    for w in study.check():
        _warnings.warn(f"design check: {w}", stacklevel=3)


async def evaluate(
    study: "Study",
    *,
    concurrency: int = 8,
    checkpoint_path: str | None = None,
    judge_checkpoint_path: str | None = None,
    on_progress: ProgressFn | None = None,
    progress: bool = True,
    _warn_design: bool = True,
) -> Evaluation:
    """Generate answers, judge them, and attribute quality — the whole pipeline.

    Shows a progress bar by default (one for answers, one for judging); pass
    ``progress=False`` to silence it, or ``on_progress`` for custom reporting.
    ``checkpoint_path`` / ``judge_checkpoint_path`` make the answer / judging phases
    crash-safe and resumable.
    """
    # Advise on thin designs before spending tokens (unless the sync wrapper already did).
    if _warn_design:
        emit_design_warnings(study)

    answers = await run_study(
        study,
        replications=study.replications,
        concurrency=concurrency,
        checkpoint_path=checkpoint_path,
        on_progress=on_progress,
        progress=progress,
    )

    questions, references = _question_and_reference_maps(study)
    ratings: Ratings | None = None
    attribution: Attribution | None = None
    if study.judge is not None and study.rubric is not None:
        ratings = await judge_results(
            answers,
            study.judge,
            study.rubric,
            repetitions=study.judge_replications,
            concurrency=concurrency,
            questions=questions,
            references=references,
            checkpoint_path=judge_checkpoint_path,
            progress=progress,
        )
        attribution = attribute(ratings)

    return Evaluation(
        study_name=study.name,
        answers=answers,
        ratings=ratings,
        attribution=attribution,
        questions=questions,
        references=references,
    )


async def preflight(study: "Study", *, concurrency: int = 8, progress: bool = False) -> Preflight:
    """Run one input through every configuration (no reps, no judging) and
    estimate the full study's cost/time."""
    answers = await run_study(study, smoke=True, concurrency=concurrency, progress=progress)
    total_cells = size(study) * max(1, len(study.dataset)) * study.replications
    plan = study.plan()
    return Preflight(
        answers=answers, estimate=estimate(answers, total_cells),
        warnings=study.check(), judge_calls=plan.get("judge_calls", 0),
    )
