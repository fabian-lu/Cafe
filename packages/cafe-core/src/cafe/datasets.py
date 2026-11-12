"""Load public benchmarks as CAFE datasets.

Each loader returns a list of items shaped for ``Study(dataset=...)``:
``{"id", "text", "reference", ...}`` — so the reference-guided judge has a gold
answer to score against. An item's ``id`` is derived from its row in the source split,
not from its position in the sample, so it names the same question whichever sample
(``n``, ``seed``, filters) you draw, and resuming a checkpoint stays safe. Needs the
``datasets`` extra (HuggingFace ``datasets``): ``pip install 'cafe-core[datasets]'``.
"""

from __future__ import annotations

import random
from typing import Any


def _quiet_hf() -> None:
    """Best-effort: silence HuggingFace's download chatter (the 'unauthenticated requests
    to the HF Hub / set HF_TOKEN' warning, progress logs). Purely cosmetic for showcasing;
    never let it break the actual load."""
    import warnings

    warnings.filterwarnings("ignore", message=r".*unauthenticated requests to the HF Hub.*")
    for mod in ("huggingface_hub", "datasets"):
        try:
            __import__(mod).logging.set_verbosity_error()
        except Exception:  # noqa: BLE001 — cosmetic only
            pass
    try:
        import logging

        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    except Exception:  # noqa: BLE001
        pass


def _load_hf(name: str, *args: Any, **kwargs: Any):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "dataset loaders need the 'datasets' extra: pip install 'cafe-core[datasets]'"
        ) from exc
    _quiet_hf()
    return load_dataset(name, *args, **kwargs)


def _sample(rows: list[Any], n: int | None, *, shuffle: bool, seed: int) -> list[Any]:
    rows = list(rows)
    if shuffle:
        random.Random(seed).shuffle(rows)
    return rows if n is None else rows[:n]


def _item_id(prefix: str, split: str, default_split: str, index: int) -> str:
    """The id of the row at ``index`` of ``split`` (counted before filtering or shuffling),
    so an id always names the same question; the split is spelled out unless it's the
    loader's default."""
    return f"{prefix}{index}" if split == default_split else f"{prefix}-{split}-{index}"


def load_truthfulqa(
    n: int | None = 20,
    *,
    split: str = "train",
    categories: list[str] | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """TruthfulQA — adversarial questions that elicit common misconceptions.

    Reference-based: each item carries the gold ``Best Answer``.

    Parameters
    ----------
    n:
        How many questions to return; ``None`` returns all of them.
    split:
        The split to load, in HuggingFace's split syntax (e.g. ``"train[:200]"``).
    categories:
        Keep only these categories, case-insensitively (e.g. ``["Misconceptions", "Law"]``).
    shuffle, seed:
        Shuffle (reproducibly, per ``seed``) before taking the first ``n``.

    Returns
    -------
    list of dict
        Items ``{id, text, reference, category}``. The ``id`` names the source row, so it
        always denotes the same question whatever ``n``, ``categories``, ``shuffle`` or
        ``seed``: ``tq<index>``, the row's index in the split before any filtering or
        shuffling, or ``tq-<split>-<index>`` for a split other than ``"train"``. A
        checkpoint written with other arguments therefore never hands a stored answer to a
        different question.
    """
    ds = _load_hf("domenicrosati/TruthfulQA", split=split)
    rows = list(enumerate(ds))  # (index in the split, row): the index names the question
    if categories:
        wanted = {c.lower() for c in categories}
        rows = [(i, r) for i, r in rows if str(r.get("Category", "")).lower() in wanted]
    rows = _sample(rows, n, shuffle=shuffle, seed=seed)
    return [
        {
            "id": _item_id("tq", split, "train", i),
            "text": r["Question"],
            "reference": r["Best Answer"],
            "category": r.get("Category"),
        }
        for i, r in rows
    ]


def load_gsm8k(
    n: int | None = 20,
    *,
    split: str = "test",
    shuffle: bool = True,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """GSM8K — grade-school math word problems with an exact numeric reference.

    Parameters
    ----------
    n:
        How many problems to return; ``None`` returns all of them.
    split:
        The split to load (``"test"`` or ``"train"``, in HuggingFace's split syntax).
    shuffle, seed:
        Shuffle (reproducibly, per ``seed``) before taking the first ``n``.

    Returns
    -------
    list of dict
        Items ``{id, text, reference}``, the reference being the final numeric answer. The
        ``id`` names the source row, so it always denotes the same problem: ``gsm<index>``,
        the row's index in the split before shuffling, or ``gsm-<split>-<index>`` for a
        split other than ``"test"``.
    """
    ds = _load_hf("openai/gsm8k", "main", split=split)
    rows = _sample(list(enumerate(ds)), n, shuffle=shuffle, seed=seed)
    return [
        {"id": _item_id("gsm", split, "test", i), "text": r["question"],
         "reference": r["answer"].split("####")[-1].strip()}
        for i, r in rows
    ]
