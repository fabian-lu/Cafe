"""Load public benchmarks as CAFE datasets.

Each loader returns a list of items shaped for ``Study(dataset=...)``:
``{"id", "text", "reference", ...}`` — so the reference-guided judge has a gold
answer to score against. Needs the ``datasets`` extra (HuggingFace ``datasets``):
``pip install 'cafe-core[datasets]'``.
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


def load_truthfulqa(
    n: int | None = 20,
    *,
    split: str = "train",
    categories: list[str] | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """TruthfulQA — adversarial questions that elicit common misconceptions.

    Reference-based: each item carries the gold ``Best Answer``. Returns items
    ``{id, text, reference, category}``. ``categories`` filters by Category
    (e.g. ``["Misconceptions", "Law"]``).
    """
    ds = _load_hf("domenicrosati/TruthfulQA", split=split)
    rows = list(ds)
    if categories:
        wanted = {c.lower() for c in categories}
        rows = [r for r in rows if str(r.get("Category", "")).lower() in wanted]
    rows = _sample(rows, n, shuffle=shuffle, seed=seed)
    return [
        {
            "id": f"tq{i}",
            "text": r["Question"],
            "reference": r["Best Answer"],
            "category": r.get("Category"),
        }
        for i, r in enumerate(rows)
    ]


def load_gsm8k(
    n: int | None = 20,
    *,
    split: str = "test",
    shuffle: bool = True,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """GSM8K — grade-school math word problems with an exact numeric reference."""
    ds = _load_hf("openai/gsm8k", "main", split=split)
    rows = _sample(list(ds), n, shuffle=shuffle, seed=seed)
    return [
        {"id": f"gsm{i}", "text": r["question"], "reference": r["answer"].split("####")[-1].strip()}
        for i, r in enumerate(rows)
    ]
