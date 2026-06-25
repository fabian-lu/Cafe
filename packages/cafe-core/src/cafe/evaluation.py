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

from dataclasses import dataclass
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
    """The complete result of evaluating a study: answers, ratings, attribution."""

    study_name: str
    answers: Results
    ratings: Ratings | None = None
    attribution: Attribution | None = None

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
        line = f"{s['n_answers']} answers · {s['n_configs']} configs · {s['n_inputs']} inputs"
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


async def evaluate(
    study: "Study",
    *,
    concurrency: int = 8,
    checkpoint_path: str | None = None,
    on_progress: ProgressFn | None = None,
    progress: bool = True,
) -> Evaluation:
    """Generate answers, judge them, and attribute quality — the whole pipeline.

    Shows a progress bar by default (one for answers, one for judging); pass
    ``progress=False`` to silence it, or ``on_progress`` for custom reporting.
    """
    answers = await run_study(
        study,
        replications=study.replications,
        concurrency=concurrency,
        checkpoint_path=checkpoint_path,
        on_progress=on_progress,
        progress=progress,
    )

    ratings: Ratings | None = None
    attribution: Attribution | None = None
    if study.judge is not None and study.rubric is not None:
        questions, references = _question_and_reference_maps(study)
        ratings = await judge_results(
            answers,
            study.judge,
            study.rubric,
            repetitions=study.judge_replications,
            concurrency=concurrency,
            questions=questions,
            references=references,
            progress=progress,
        )
        attribution = attribute(ratings)

    return Evaluation(
        study_name=study.name,
        answers=answers,
        ratings=ratings,
        attribution=attribution,
    )


async def preflight(study: "Study", *, concurrency: int = 8, progress: bool = False) -> Preflight:
    """Run one input through every configuration (no reps, no judging) and
    estimate the full study's cost/time."""
    answers = await run_study(study, smoke=True, concurrency=concurrency, progress=progress)
    total_cells = size(study) * max(1, len(study.dataset)) * study.replications
    return Preflight(answers=answers, estimate=estimate(answers, total_cells))
