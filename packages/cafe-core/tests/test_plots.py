"""The plot menu renders without error (headless Agg backend)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import cafe  # noqa: E402
from cafe.evaluation import Evaluation  # noqa: E402
from cafe.execution.results import Observation, Results  # noqa: E402
from cafe.judging.ratings import Rating, Ratings  # noqa: E402


def _evaluation() -> Evaluation:
    score = {("120b", "cot"): 5, ("120b", "concise"): 4, ("20b", "cot"): 3, ("20b", "concise"): 4}
    obs, items = [], []
    for q in range(6):
        for m in ("120b", "20b"):
            for p in ("cot", "concise"):
                cfg = {"model": m, "prompt": p}
                o = Observation(config=cfg, input_id=f"q{q}", rep=0, output="a", elapsed_s=1.0 + q * 0.1)
                obs.append(o)
                s = max(1, min(5, score[(m, p)] - (q % 2)))
                items.append(Rating(obs_key=o.key(), config=cfg, input_id=f"q{q}",
                                    rep=0, judge_rep=0, value=s, value_numeric=s))
    ans = Results(study_name="t", factors=["model", "prompt"], observations=obs)
    rt = Ratings(rubric=cafe.ANSWER_QUALITY_1_5, judge_model="j",
                 factors=["model", "prompt"], items=items)
    return Evaluation(study_name="t", answers=ans, ratings=rt, attribution=cafe.attribute(rt))


@pytest.mark.parametrize("kind", ["marginals", "interaction", "configs", "distribution", "effects"])
def test_each_plot_kind_renders(kind):
    ax = _evaluation().plot(kind)
    assert ax is not None
    plt.close("all")


def test_dashboard_renders():
    fig = _evaluation().plot()
    assert fig is not None and len(fig.axes) >= 4
    plt.close("all")


def test_unknown_kind_errors():
    with pytest.raises(ValueError, match="unknown plot"):
        _evaluation().plot("nope")
