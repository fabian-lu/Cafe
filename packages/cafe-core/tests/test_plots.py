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


# the same model from two providers, and two prompts that share their first 16 characters
_MODELS = ["openai/gpt-4o-mini", "azure/gpt-4o-mini"]
_PROMPTS = ["Answer the question concisely.", "Answer the question step by step."]


def _colliding_evaluation() -> Evaluation:
    obs, items = [], []
    for q in range(6):
        for m in _MODELS:
            for p in _PROMPTS:
                cfg = {"model": m, "prompt": p}
                o = Observation(config=cfg, input_id=f"q{q}", rep=0, output="a", elapsed_s=1.0)
                obs.append(o)
                s = 2 + 2 * m.startswith("openai") + p.endswith("step.") - (q % 2)
                items.append(Rating(obs_key=o.key(), config=cfg, input_id=f"q{q}",
                                    rep=0, judge_rep=0, value=s, value_numeric=s))
    ans = Results(study_name="t", factors=["model", "prompt"], observations=obs)
    rt = Ratings(rubric=cafe.ANSWER_QUALITY_1_5, judge_model="j",
                 factors=["model", "prompt"], items=items)
    return Evaluation(study_name="t", answers=ans, ratings=rt, attribution=cafe.attribute(rt))


def _texts(labels) -> list[str]:
    return [t.get_text() for t in labels]


def test_levels_whose_short_labels_collide_are_told_apart():
    """Every level of these two factors used to share its tick label with the other level
    (gpt-4o-mini twice, 'Answer the ques…' twice), so the bars and lines couldn't be told apart."""
    ev = _colliding_evaluation()
    marginals = _texts(ev.plot("marginals").get_xticklabels())
    assert marginals == ["azure/gpt-4o-mini", "openai/gpt-4o-mini",
                         "Answer the question concisely.", "Answer the question step by ste…"]

    ax = ev.plot("interaction", factors=["model", "prompt"])
    assert _texts(ax.get_xticklabels()) == ["azure/gpt-4o-mini", "openai/gpt-4o-mini"]
    assert len(set(_texts(ax.get_legend().get_texts()))) == 2

    for kind, ticks in (("configs", "get_yticklabels"), ("distribution", "get_xticklabels")):
        labels = _texts(getattr(ev.plot(kind), ticks)())
        assert len(labels) == 4 and len(set(labels)) == 4, (kind, labels)

    effects = _texts(ev.plot("effects").get_yticklabels())
    assert "model: azure/gpt-4o-mini vs openai/gpt-4o-mini" in effects
    for label in effects:   # was "model: gpt-4o-mini vs gpt-4o-mini"
        a, b = label.split(": ", 1)[1].split(" vs ")
        assert a != b
    plt.close("all")


def test_level_labels_stay_short_unless_they_collide():
    from cafe.stats.plots import _labels

    assert _labels(["ollama_cloud/gpt-oss:120b", "ollama_cloud/gpt-oss:20b"]) == \
        ["gpt-oss:120b", "gpt-oss:20b"]
    assert _labels(["openai/gpt-4o-mini", "azure/gpt-4o-mini", "openai/o3"]) == \
        ["openai/gpt-4o-mini", "azure/gpt-4o-mini", "o3"]
    # still the same once more of the text is shown: numbered
    assert _labels([1, "1"]) == ["1 (1)", "1 (2)"]
    assert _labels(["x" * 40 + "a", "x" * 40 + "b"]) == ["xxxxxxxxxxxxxxx… (1)", "xxxxxxxxxxxxxxx… (2)"]
