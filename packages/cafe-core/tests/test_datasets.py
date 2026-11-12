"""Benchmark loaders: an item's id names its source row, whichever sample is drawn.

The HuggingFace download is stubbed, so these tests run offline.
"""

import random
import warnings

import pytest

import cafe
import cafe.datasets as ds

TQ_ROWS = [{"Question": f"Question {i}?", "Best Answer": f"ref {i}",
            "Category": "Law" if i % 2 else "Health"} for i in range(40)]
GSM_ROWS = {split: [{"question": f"{split} problem {i}?", "answer": f"work #### {i}"}
                    for i in range(30)] for split in ("test", "train")}


@pytest.fixture
def stub_hf(monkeypatch):
    def load(name, *args, split):
        return TQ_ROWS if name == "domenicrosati/TruthfulQA" else GSM_ROWS[split]

    monkeypatch.setattr(ds, "_load_hf", load)


def test_truthfulqa_ids_name_the_source_row(stub_hf):
    question = {f"tq{i}": r["Question"] for i, r in enumerate(TQ_ROWS)}
    samples = [ds.load_truthfulqa(n=5, seed=seed) for seed in (0, 1, 2)]
    samples += [ds.load_truthfulqa(n=5, categories=["Law"]), ds.load_truthfulqa(n=None)]
    for items in samples:
        assert all(it["text"] == question[it["id"]] for it in items)

    assert [it["id"] for it in ds.load_truthfulqa(n=3, shuffle=False)] == ["tq0", "tq1", "tq2"]
    law = ds.load_truthfulqa(n=3, shuffle=False, categories=["law"])
    assert [it["id"] for it in law] == ["tq1", "tq3", "tq5"]  # not renumbered after filtering

    # the sample itself is unchanged: the same questions, in the same order, per seed
    rows = list(TQ_ROWS)
    random.Random(1).shuffle(rows)
    assert [it["text"] for it in samples[1]] == [r["Question"] for r in rows[:5]]


def test_gsm8k_ids_name_the_source_row_and_split(stub_hf):
    question = {f"gsm{i}": r["question"] for i, r in enumerate(GSM_ROWS["test"])}
    question |= {f"gsm-train-{i}": r["question"] for i, r in enumerate(GSM_ROWS["train"])}
    for split in ("test", "train"):
        for seed in (0, 1):
            items = ds.load_gsm8k(n=5, seed=seed, split=split)
            assert all(it["text"] == question[it["id"]] for it in items)
    assert [it["id"] for it in ds.load_gsm8k(n=2, shuffle=False, split="train")] == [
        "gsm-train-0", "gsm-train-1"]


async def test_resuming_with_another_sample_never_reuses_an_answer_for_another_question(
    stub_hf, tmp_path
):
    # Positional ids (tq0, tq1, ...) named different questions in different samples, so a
    # checkpoint reused after changing the seed handed old answers to the new questions.
    async def system(config, item):
        return f"answer to {item['text']}"

    ckpt = str(tmp_path / "run.jsonl")

    def study(seed):
        return cafe.Study(name="tq", system=system, factors=[cafe.Factor("m", ["x"])],
                          dataset=ds.load_truthfulqa(n=3, seed=seed))

    await cafe.run_study(study(0), checkpoint_path=ckpt)
    resumed = study(1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = await cafe.run_study(resumed, checkpoint_path=ckpt)
    question = {it["id"]: it["text"] for it in resumed.dataset}
    assert all(o.output == f"answer to {question[o.input_id]}" for o in results.observations)
    # the other sample's answers are dropped as stale, not carried forward
    assert any("no longer in the study's design" in str(w.message) for w in caught)
