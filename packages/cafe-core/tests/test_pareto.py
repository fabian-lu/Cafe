"""Pareto frontier over quality / cost / latency."""

import cafe
from cafe.evaluation import Evaluation
from cafe.execution.results import Observation, Results
from cafe.judging.ratings import Rating, Ratings


def _evaluation():
    """Three configs: A high-quality/expensive, B low-cost/lower-quality, C dominated."""
    specs = {
        "A": {"quality": 5, "cost": 0.10, "latency": 5.0},
        "B": {"quality": 4, "cost": 0.01, "latency": 1.0},
        "C": {"quality": 3, "cost": 0.11, "latency": 6.0},  # worse than A on everything
    }
    obs, ratings = [], []
    for name, s in specs.items():
        config = {"sys": name}
        for i in range(2):  # two items each
            o = Observation(
                config=config, input_id=f"q{i}", rep=0, output="ans",
                elapsed_s=s["latency"], metadata={"cost_usd": s["cost"], "tokens": 100},
            )
            obs.append(o)
            ratings.append(Rating(
                obs_key=o.key(), config=config, input_id=f"q{i}", rep=0, judge_rep=0,
                value=s["quality"], value_numeric=s["quality"],
            ))
    answers = Results(study_name="t", factors=["sys"], observations=obs)
    rt = Ratings(rubric=cafe.ANSWER_QUALITY_1_5, judge_model="m", factors=["sys"], items=ratings)
    return Evaluation(study_name="t", answers=answers, ratings=rt)


def test_pareto_identifies_frontier_and_dominated():
    res = cafe.pareto(_evaluation())
    optimal = {r["label"] for r in res.frontier}
    assert "sys=A" in optimal      # best quality
    assert "sys=B" in optimal      # cheapest + fastest
    assert "sys=C" not in optimal  # dominated by A on all objectives


def test_pareto_objectives_drop_constant():
    res = cafe.pareto(_evaluation())
    # all of quality/cost/latency vary here, so all are used
    assert set(res.objectives) == {"quality", "cost", "latency"}


def test_pareto_values_are_means():
    rows = {r["label"]: r for r in cafe.pareto(_evaluation()).rows}
    assert rows["sys=A"]["quality"] == 5.0
    assert rows["sys=B"]["cost"] == 0.01


def test_pareto_plot_returns_axes():
    import matplotlib
    matplotlib.use("Agg")
    ax = cafe.pareto(_evaluation()).plot(x="cost", y="quality")
    assert ax is not None


def test_pareto_quality_collapses_judge_replications():
    """Judge reps must be averaged per answer first (the analysis-frame rule) —
    otherwise answers with more surviving verdicts get more weight and pareto()
    disagrees with attribute() on the same data."""
    config = {"sys": "a"}
    obs, items = [], []
    # q0: judged 3x [5,5,5]; q1: one usable verdict [1] → per-answer means 5 and 1
    o0 = Observation(config=config, input_id="q0", rep=0, output="x",
                     metadata={"cost_usd": 0.1})
    o1 = Observation(config=config, input_id="q1", rep=0, output="y",
                     metadata={"cost_usd": 0.1})
    obs += [o0, o1]
    for jr in range(3):
        items.append(Rating(obs_key=o0.key(), config=config, input_id="q0", rep=0,
                            judge_rep=jr, value=5, value_numeric=5))
    items.append(Rating(obs_key=o1.key(), config=config, input_id="q1", rep=0,
                        judge_rep=0, value=1, value_numeric=1))
    ev = Evaluation(
        study_name="t",
        answers=Results(study_name="t", factors=["sys"], observations=obs),
        ratings=Ratings(rubric=cafe.ANSWER_QUALITY_1_5, judge_model="m",
                        factors=["sys"], items=items),
    )
    rows = {r["label"]: r for r in cafe.pareto(ev).rows}
    assert rows["sys=a"]["quality"] == 3.0   # (5 + 1) / 2 answers, not (5+5+5+1)/4 = 4.0


def test_pareto_skips_missing_cost_instead_of_counting_zero():
    config = {"sys": "a"}
    o0 = Observation(config=config, input_id="q0", rep=0, output="x",
                     metadata={"cost_usd": 0.30})
    o1 = Observation(config=config, input_id="q1", rep=0, output="y", metadata={})  # unreported
    items = [Rating(obs_key=o.key(), config=config, input_id=o.input_id, rep=0,
                    judge_rep=0, value=4, value_numeric=4) for o in (o0, o1)]
    ev = Evaluation(
        study_name="t",
        answers=Results(study_name="t", factors=["sys"], observations=[o0, o1]),
        ratings=Ratings(rubric=cafe.ANSWER_QUALITY_1_5, judge_model="m",
                        factors=["sys"], items=items),
    )
    rows = {r["label"]: r for r in cafe.pareto(ev).rows}
    assert rows["sys=a"]["cost"] == 0.30     # missing value skipped, mean not diluted to 0.15
