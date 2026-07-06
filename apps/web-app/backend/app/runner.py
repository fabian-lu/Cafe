"""Run a study: build a cafe.Study from the DB rows + the discovered pipeline, execute it as an
asyncio background task, stream progress, and cache the results. All the science is cafe-core's.
"""

from __future__ import annotations

import asyncio
import traceback
from typing import Any

import cafe

from app.db import SessionLocal
from app import models
from app.pipeline_loader import get_pipeline

# In-memory progress, read by the SSE endpoint. {study_id: {"phase","done","total","status"}}
PROGRESS: dict[int, dict[str, Any]] = {}


def _build_rubric(row: models.Rubric) -> cafe.Rubric:
    levels = [cafe.Level(int(l["value"]), l.get("label", ""), l.get("description", ""))
              for l in row.levels]
    return cafe.Rubric(
        name=row.name, scale_type=row.scale_type, levels=levels,
        instruction=row.instruction or "Judge the answer.",
        prompt_template=row.prompt_template or None,
    )


def _coerce_levels(levels: list[Any]) -> list[Any]:
    # None stays None (skip level); numeric-looking strings → numbers.
    out = []
    for v in levels:
        if v is None:
            out.append(None)
        elif isinstance(v, str) and v.strip() != "" and _is_num(v):
            out.append(float(v) if "." in v else int(v))
        else:
            out.append(v)
    return out


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


async def _load_study_objects(study_id: int):
    """Assemble the cafe.Study (+ dataset/rubric/judge) from the DB rows."""
    async with SessionLocal() as db:
        study = await db.get(models.Study, study_id)
        if study is None:
            raise ValueError(f"study {study_id} not found")
        dataset = await db.get(models.Dataset, study.dataset_id) if study.dataset_id else None
        rubric_row = await db.get(models.Rubric, study.rubric_id) if study.rubric_id else None
        # snapshot the plain fields we need (avoid lazy-load after the session closes)
        name = study.name
        pipeline_name = getattr(study, "pipeline", None) or "pipeline"
        factors_raw = list(study.factors)
        judge_model = study.judge_model
        reps = study.replications
        items = list(dataset.items) if dataset else []
        rubric = _build_rubric(rubric_row) if rubric_row else None
        judge_preset = getattr(rubric_row, "preset", None) or "reference_qa"
        judge_system = getattr(rubric_row, "system_prompt", None)
        concurrency = max(1, getattr(study, "concurrency", 8) or 8)

    pipe = get_pipeline(pipeline_name)
    factors = [cafe.Factor(f["name"], _coerce_levels(f["levels"])) for f in factors_raw]
    judge = None
    if judge_model and rubric:
        judge = cafe.LLMJudge(model=judge_model, preset=judge_preset, system_prompt=judge_system)

    study_obj = cafe.Study(
        name=name, system=pipe, factors=factors, dataset=items,
        rubric=rubric, judge=judge, replications=reps,
    )
    return study_obj, concurrency


def _results_payload(ev) -> dict[str, Any]:
    """Serialise a cafe Evaluation into JSON the Results page draws."""
    import numpy as np
    import pandas as pd

    def df_records(df):
        if df is None:
            return []
        return df.replace({np.nan: None}).to_dict("records")

    out: dict[str, Any] = {
        "report": ev.report() if ev.attribution is not None else "",
        "overall_mean": ev.overall_mean,
        "factors": _factor_keys(ev),
        "records": [], "marginal_means": [], "effects": [], "pareto": None,
        "config_means": [], "best_config": None, "pairwise_d": [],
        "variance_components": None, "clmm": None, "logistic": None, "rubric": None,
        "effects_model": None, "r_squared": None,
        "timing": _timing(ev.answers),
    }
    try:
        out["records"] = [
            {k: r.get(k) for k in ("input_id", "question", "reference", "answer", "verdict",
                                   "reasoning", "cost_usd", "tokens", "elapsed_s", *(_factor_keys(ev)))}
            for r in ev.records()
        ]
    except Exception:  # noqa: BLE001
        pass
    try:
        out["marginal_means"] = df_records(ev.marginal_means)
    except Exception:  # noqa: BLE001
        pass
    # Per-configuration ranking + best config (descriptive layer).
    try:
        attr = ev.attribution
        if attr is not None:
            out["config_means"] = [dict(c) for c in attr.config_means]
            out["best_config"] = dict(attr.best_config) if attr.best_config else None
    except Exception:  # noqa: BLE001
        pass
    # Inferential layer: F/p/η² terms, Cohen's d effect sizes, variance components.
    try:
        eff = ev.effects
        if eff is not None:
            out["effects"] = df_records(eff.to_df())
            out["pairwise_d"] = [dict(d) for d in eff.pairwise_d]
            out["variance_components"] = dict(eff.variance_components) if eff.variance_components else None
            out["effects_model"] = eff.model
            # R² — total share of verdict variance the model explains (observed = fitted + residual)
            if eff.residuals and eff.fitted and len(eff.residuals) == len(eff.fitted):
                obs = [f + r for f, r in zip(eff.fitted, eff.residuals)]
                mean = sum(obs) / len(obs)
                ss_tot = sum((o - mean) ** 2 for o in obs)
                ss_res = sum(r * r for r in eff.residuals)
                out["r_squared"] = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None
    except Exception:  # noqa: BLE001
        pass
    # The rubric the verdicts were scored on (scale + levels) — labels the distribution axis and
    # decides which model is the statistically-correct one to fit below.
    scale = None
    try:
        rb = ev.ratings.rubric if ev.ratings else None
        if rb is not None:
            scale = str(getattr(rb.scale_type, "value", rb.scale_type))
            out["rubric"] = {
                "name": rb.name, "scale_type": scale,
                "levels": [{"value": lv.value, "label": lv.label} for lv in rb.levels],
            }
    except Exception:  # noqa: BLE001
        pass
    # The scale-correct extra model: ordinal → CLMM, binary → logistic, numeric → the linear model
    # above IS the correct one (no extra fit). Only fit the matching one (each is an R subprocess).
    if scale == "ordinal":
        try:
            clmm = ev.clmm
            if clmm is not None:
                out["clmm"] = {
                    "available": bool(clmm.available), "reason": clmm.reason,
                    "n_obs": clmm.n_obs, "formula": clmm.formula, "log_lik": clmm.log_lik,
                    "coefficients": [dict(c) for c in clmm.coefficients],
                    "thresholds": [dict(t) for t in clmm.thresholds],
                }
        except Exception:  # noqa: BLE001
            pass
    elif scale == "binary":
        try:
            log = ev.logistic
            if log is not None:
                out["logistic"] = {
                    "available": bool(log.available), "reason": log.reason,
                    "n_obs": log.n_obs, "formula": log.formula,
                    "terms": [dict(t) for t in log.terms],
                }
        except Exception:  # noqa: BLE001
            pass
    try:
        from cafe.stats.pareto import pareto as _pareto
        pf = _pareto(ev)
        out["pareto"] = {"objectives": list(pf.objectives),
                         "rows": [dict(r) for r in getattr(pf, "rows", [])]}
    except Exception:  # noqa: BLE001
        pass
    return out


def _factor_keys(ev) -> list[str]:
    try:
        return list(ev.ratings.factors) if ev.ratings else []
    except Exception:  # noqa: BLE001
        return []


def _timing(answers) -> dict[str, Any]:
    """How long the run took, straight from the observations cafe already tracked: total compute
    (sum of per-cell time), wall-clock span (from ``started_at`` + ``elapsed_s``), and a per-stage
    breakdown aggregated from each answer's execution ``trace``."""
    import datetime as _dt

    obs = [o for o in answers.observations if o.ok]
    times = [o.elapsed_s for o in obs if o.elapsed_s is not None]
    total_compute_s = sum(times) if times else 0.0

    starts, ends = [], []
    for o in obs:
        if o.started_at and o.elapsed_s is not None:
            try:
                st = _dt.datetime.fromisoformat(str(o.started_at))
                starts.append(st)
                ends.append(st + _dt.timedelta(seconds=o.elapsed_s))
            except (ValueError, TypeError):
                pass
    wall_s = (max(ends) - min(starts)).total_seconds() if starts and ends else None

    stage_s: dict[str, float] = {}
    for o in obs:
        for step in (o.metadata or {}).get("trace", []):
            stg = step.get("stage")
            if stg is not None:
                stage_s[stg] = stage_s.get(stg, 0.0) + (step.get("elapsed_s") or 0.0)

    return {
        "n_answers": len(obs),
        "total_compute_s": round(total_compute_s, 2),
        "wall_s": round(wall_s, 2) if wall_s is not None else None,
        "mean_cell_s": round(total_compute_s / len(times), 3) if times else None,
        "per_stage_s": {k: round(v, 2) for k, v in stage_s.items()},
    }


async def estimate_study(study_id: int) -> dict[str, Any]:
    """Preflight the study (one input through every configuration, no judging) → cost/time estimate
    for the full run + design warnings. Makes real calls, so it takes a moment (hence the UI spinner)."""
    from cafe.evaluation import preflight

    study_obj, concurrency = await _load_study_objects(study_id)
    pf = await preflight(study_obj, concurrency=concurrency, progress=False)
    return {"estimate": pf.estimate, "warnings": pf.warnings, "judge_calls": pf.judge_calls}


async def run_study_task(study_id: int) -> None:
    """Background task: execute the study and cache its results."""
    PROGRESS[study_id] = {"phase": "starting", "done": 0, "total": 0, "status": "running"}
    try:
        import time as _time
        run_started = _time.monotonic()
        study_obj, concurrency = await _load_study_objects(study_id)
        total = len(cafe.design.generate(study_obj)) * max(1, len(study_obj.dataset)) * study_obj.replications
        PROGRESS[study_id].update({"phase": "answers", "total": total})

        # Run the two phases explicitly so we can report BOTH answer and judging progress
        # (cafe.evaluate only surfaces the answer phase) — matches the notebook's two progress bars.
        from cafe.execution import run_study
        from cafe.judging import judge_results
        from cafe.stats import attribute
        from cafe.evaluation import Evaluation, _question_and_reference_maps

        def ans_cb(_obs, done, tot):
            PROGRESS[study_id].update({"phase": "answers", "done": done, "total": tot})

        answers = await run_study(study_obj, replications=study_obj.replications,
                                  concurrency=concurrency, on_progress=ans_cb, progress=False)

        questions, references = _question_and_reference_maps(study_obj)
        ratings = attribution = None
        if study_obj.judge is not None and study_obj.rubric is not None:
            def judge_cb(_r, done, tot):
                PROGRESS[study_id].update({"phase": "judging", "done": done, "total": tot})
            ratings = await judge_results(
                answers, study_obj.judge, study_obj.rubric,
                repetitions=study_obj.judge_replications, concurrency=concurrency,
                questions=questions, references=references, on_progress=judge_cb, progress=False)
            attribution = attribute(ratings)

        ev = Evaluation(study_name=study_obj.name, answers=answers, ratings=ratings,
                        attribution=attribution, questions=questions, references=references)
        PROGRESS[study_id].update({"phase": "saving", "done": total, "total": total})

        payload = _results_payload(ev)
        # true end-to-end wall clock (answers + judging), the "total time it took to run"
        payload["timing"]["run_wall_s"] = round(_time.monotonic() - run_started, 2)
        async with SessionLocal() as db:
            existing = await db.get(models.Study, study_id)
            if existing:
                existing.status = "done"
                existing.progress = 100
            # upsert the cached result
            from sqlalchemy import select
            res = (await db.execute(select(models.StudyResult).where(
                models.StudyResult.study_id == study_id))).scalar_one_or_none()
            if res is None:
                db.add(models.StudyResult(study_id=study_id, payload=payload))
            else:
                res.payload = payload
            await db.commit()
        PROGRESS[study_id].update({"phase": "done", "status": "done"})
    except Exception as exc:  # noqa: BLE001
        PROGRESS[study_id] = {"phase": "error", "status": "failed", "error": str(exc)}
        traceback.print_exc()
        async with SessionLocal() as db:
            s = await db.get(models.Study, study_id)
            if s:
                s.status = "failed"
                await db.commit()


def launch(study_id: int) -> None:
    """Fire-and-forget the background task."""
    asyncio.create_task(run_study_task(study_id))
