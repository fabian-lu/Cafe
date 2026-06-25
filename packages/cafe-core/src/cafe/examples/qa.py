"""A real example system: question answering via any LiteLLM-supported model.

Two factors that actually matter:

- ``model``  — which LLM answers (e.g. ``gpt-oss:120b`` vs ``gpt-oss:20b`` on
  Ollama Cloud, or ``gpt-4o``, ``anthropic/claude-...``)
- ``prompt`` — the prompting strategy (concise vs chain-of-thought)

The system just calls :func:`cafe.llm.complete`, so it works with any provider.
Requires the ``llm`` extra and the relevant API key (see ``.env.example``).
"""

from __future__ import annotations

from typing import Any

from cafe.llm import complete

_PROMPTS = {
    "concise": "Answer the question correctly and concisely.",
    "cot": (
        "Reason step by step. Then end with a single final line starting with "
        "'Answer:' giving the concise answer."
    ),
}


async def qa_system(config: dict[str, Any], item: Any) -> dict[str, Any]:
    """Answer ``item`` with the configured model + prompt strategy."""
    question = item["text"] if isinstance(item, dict) and "text" in item else str(item)
    strategy = config.get("prompt", "concise")
    answer = await complete(
        config["model"],
        messages=[
            {"role": "system", "content": _PROMPTS.get(strategy, _PROMPTS["concise"])},
            {"role": "user", "content": question},
        ],
        temperature=float(config.get("temperature", 0.0)),
    )
    return {"output": answer}


# A small, checkable set with reference answers for the judge.
DEFAULT_QUESTIONS = [
    {"id": "q1", "text": "What is the capital of Australia?", "reference": "Canberra."},
    {"id": "q2", "text": "What is 17 multiplied by 23?", "reference": "391."},
    {"id": "q3", "text": "In which year did the Berlin Wall fall?", "reference": "1989."},
]


def build_qa_study(
    models: tuple[str, ...] = ("ollama_cloud/gpt-oss:120b", "ollama_cloud/gpt-oss:20b"),
    dataset: list[dict[str, Any]] | None = None,
    replications: int = 1,
):
    """A 2-factor QA study (model × prompt), judged on a 1–5 quality rubric."""
    from cafe.judging import LLMJudge
    from cafe.judging.rubric import ANSWER_QUALITY_1_5
    from cafe.study import Factor, FactorType, Study

    return Study(
        name="qa",
        system=qa_system,
        factors=[
            Factor("model", list(models), FactorType.categorical),
            Factor("prompt", ["concise", "cot"], FactorType.categorical),
        ],
        dataset=dataset or DEFAULT_QUESTIONS,
        rubric=ANSWER_QUALITY_1_5,
        judge=LLMJudge(models[0]),
        replications=replications,
    )
