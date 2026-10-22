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


def test_pareto_quality_counts_each_answer_once():
    """judge_replications=3: sys=a has one answer judged [5,5,5] and one with a single usable
    verdict [1] (two judge calls failed); sys=b costs twice as much and is judged [4,3,4] /
    [3,4,3]. Per answer, a scores 5 and 1 (mean 3.0), b 4 and 3 (3.5) — as attribute() says.
    Pooling every verdict instead gave a 4.0, so a wrongly dominated b."""
    import pytest

    verdicts = {"a": {"q0": [5, 5, 5], "q1": [1, None, None]},
                "b": {"q0": [4, 3, 4], "q1": [3, 4, 3]}}
    obs, items = [], []
    for name, per_q in verdicts.items():
        for q, vs in per_q.items():
            o = Observation(config={"sys": name}, input_id=q, rep=0, output="ans", elapsed_s=1.0,
                            metadata={"cost_usd": 0.01 if name == "a" else 0.02, "tokens": 100})
            obs.append(o)
            items.extend(
                Rating(obs_key=o.key(), config={"sys": name}, input_id=q, rep=0, judge_rep=jr,
                       value=v, value_numeric=v, error=None if v is not None else "judge failed")
                for jr, v in enumerate(vs)
            )
    rt = Ratings(rubric=cafe.ANSWER_QUALITY_1_5, judge_model="m", factors=["sys"], items=items)
    ev = Evaluation(study_name="t", answers=Results(study_name="t", factors=["sys"],
                                                    observations=obs), ratings=rt)

    res = cafe.pareto(ev)
    quality = {r["config"]["sys"]: r["quality"] for r in res.rows}
    attributed = {c["config"]["sys"]: c["mean"] for c in cafe.attribute(rt).config_means}
    assert quality == pytest.approx({"a": 3.0, "b": 3.5}) == attributed
    assert {r["label"] for r in res.frontier} == {"sys=a", "sys=b"}  # cheaper vs better


def test_pareto_plot_returns_axes():
    import matplotlib
    matplotlib.use("Agg")
    ax = cafe.pareto(_evaluation()).plot(x="cost", y="quality")
    assert ax is not None


def _metadata_evaluation(meta_for):
    """Configs a–d, four answers each, all scored 4; ``meta_for(config, i)`` -> metadata."""
    obs, ratings = [], []
    for name in ("a", "b", "c", "d"):
        for i in range(4):
            o = Observation(config={"sys": name}, input_id=f"q{i}", rep=0, output="ans",
                            elapsed_s=1.0, metadata=meta_for(name, i))
            obs.append(o)
            ratings.append(Rating(obs_key=o.key(), config={"sys": name}, input_id=f"q{i}",
                                  rep=0, judge_rep=0, value=4, value_numeric=4))
    answers = Results(study_name="t", factors=["sys"], observations=obs)
    rt = Ratings(rubric=cafe.ANSWER_QUALITY_1_5, judge_model="m", factors=["sys"], items=ratings)
    return Evaluation(study_name="t", answers=answers, ratings=rt)


def test_pareto_missing_cost_and_tokens_are_not_zero():
    """a reports $0.04 / 400 tokens on every answer; b has the same cost but drops the
    metadata on half its answers; c never reports it; d reports a genuine 0 (free)."""
    import pytest

    def meta(name, i):
        if name == "a" or (name == "b" and i % 2 == 0):
            return {"cost_usd": 0.04, "tokens": 400}
        return {"cost_usd": 0.0, "tokens": 0} if name == "d" else {}

    res = cafe.pareto(_metadata_evaluation(meta))
    rows = {r["config"]["sys"]: r for r in res.rows}
    assert rows["b"]["cost"] == rows["a"]["cost"] == 0.04     # not diluted to 0.02
    assert rows["b"]["tokens"] == 400.0
    assert rows["c"]["cost"] is None and rows["c"]["tokens"] is None   # unknown, not free
    assert rows["d"]["cost"] == 0.0 and rows["d"]["tokens"] == 0.0     # a reported 0 counts
    # an objective some config doesn't report can't decide dominance: c doesn't win on "cost"
    assert "cost" not in res.objectives and "tokens" not in res.objectives
    assert {r["config"]["sys"] for r in res.frontier} == {"a", "b", "c", "d"}
    with pytest.raises(ValueError, match="not reported"):
        cafe.pareto(_metadata_evaluation(meta), objectives=["quality", "cost"])
    with pytest.raises(ValueError, match="not reported"):
        res.plot(x="cost")
