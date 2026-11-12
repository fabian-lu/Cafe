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


def _two_factor_evaluation(a: str, a_levels, b: str, b_levels, score) -> Evaluation:
    obs, items = [], []
    for q in range(4):
        for la in a_levels:
            for lb in b_levels:
                cfg = {a: la, b: lb}
                o = Observation(config=cfg, input_id=f"q{q}", rep=0, output="a", elapsed_s=1.0)
                obs.append(o)
                s = score(la, lb)
                items.append(Rating(obs_key=o.key(), config=cfg, input_id=f"q{q}",
                                    rep=0, judge_rep=0, value=s, value_numeric=s))
    ans = Results(study_name="t", factors=[a, b], observations=obs)
    rt = Ratings(rubric=cafe.ANSWER_QUALITY_1_5, judge_model="j", factors=[a, b], items=items)
    return Evaluation(study_name="t", answers=ans, ratings=rt, attribution=cafe.attribute(rt))


def test_interaction_plot_orders_numeric_levels_by_value():
    """Sorting the levels as text drew top_k 5, 10, 20 as 10, 20, 5: a rising trend became an
    inverted V, right next to a marginals panel that orders them 5, 10, 20."""
    rises = {5: 2, 10: 3, 20: 4}
    ev = _two_factor_evaluation("top_k", [20, 5, 10], "model", ["small", "large"],
                                lambda k, m: rises[k] + (m == "large"))
    ax = ev.plot("interaction", factors=["top_k", "model"])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["5", "10", "20"]
    for line in ax.get_lines():
        assert list(line.get_ydata()) == sorted(line.get_ydata())   # still rising
    marginals = ev.plot("marginals")
    assert [t.get_text() for t in marginals.get_xticklabels()][:3] == ["5", "10", "20"]

    legend = ev.plot("interaction", factors=["model", "top_k"]).get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["5", "10", "20"]
    plt.close("all")


@pytest.mark.parametrize(
    "levels, shown",
    [
        ([0.5, 10, 2], ["0.5", "2.0", "10.0"]),   # a float column
        (["auto", 10, 2], ["2", "10", "auto"]),   # numbers first, then text
        ([True, False], ["False", "True"]),
    ],
)
def test_interaction_plot_level_order_matches_the_marginals(levels, shown):
    ev = _two_factor_evaluation("knob", levels, "model", ["a", "b"],
                                lambda k, m: 3 + (m == "b"))
    ax = ev.plot("interaction", factors=["knob", "model"])
    assert [t.get_text() for t in ax.get_xticklabels()] == shown
    marginals = ev.plot("marginals")
    assert [t.get_text() for t in marginals.get_xticklabels()][:len(shown)] == shown
    plt.close("all")
