# Case studies

Three complete CAFE studies, each a self-contained folder with the system under test, a notebook
that runs the study, the data, and the saved results. Together they show everything CAFE does.

| | Study | The question | Factors | Scored by |
|---|---|---|---|---|
| 1 | [Question answering](1_question_answering/) (HotpotQA) | What drives answer quality in a RAG pipeline: retrieval, model, or both? | `retrieval` × `model` × `placebo` | LLM judge, 0–2 |
| 2 | [Support chatbot](2_support_chatbot/) (Bitext) | Is a cheaper model good enough? Does a support handbook in the prompt help? | `system_prompt` × `model` | LLM judge, 1–5 and 1–3 |
| 3 | [Invoice processing](3_invoice_processing/) | Can a small model extract booking data? Do the PDF conversion and a checking step matter? | `model` × `pdf_text` × `prompting` | code, ready to book yes/no |

## Which study shows what

| CAFE feature | 1 | 2 | 3 |
|---|---|---|---|
| Pipeline stages as factors, model as a factor | ✓ | ✓ | ✓ |
| Interactions (one factor's effect depends on another) | ✓ | ✓ | ✓ |
| Placebo factor (CAFE finds no effect where there is none) | ✓ | | |
| Composed mode: a pipeline built from CAFE stages, also runnable in the web app | ✓ | ✓ | |
| Black-box mode: any existing system as `run(config, item)` | | | ✓ |
| LLM judge with an ordinal rubric | ✓ | ✓ | |
| Several rubrics on the same answers (dimensions) | | ✓ | |
| A judge written in plain code, binary rubric | | | ✓ |
| Linear model (attribution: F, partial η²), effect sizes | ✓ | ✓ | ✓ |
| Ordinal mixed model (CLMM) with per-factor tests | ✓ | ✓ | |
| Logistic mixed model (GLMM) with per-factor tests | | | ✓ |
| Cost/quality Pareto front, energy per configuration | ✓ | ✓ | ✓ |
| Question filters (metadata) in the web app | ✓ | ✓ | ✓ |

## Running a study

From the repository root, with API keys for Ollama Cloud and OpenRouter in `.env`:

```bash
pip install -e "packages/cafe-core[all]" pypdf
cd case_studies/<study folder>
jupyter lab
```

Open the notebook and run it top to bottom. Each notebook ends by saving its results to
`results/`; in the web app, **Studies → Import study** turns that file into a browsable study.

## Models, prices and energy

All three studies use the same model names. The larger models run on Ollama Cloud; Gemma 3 4B runs
through OpenRouter because Ollama Cloud retired it. For the cost/quality analysis every model is
charged at its OpenRouter list price (October 2026). The energy figures are estimates per 1,000
tokens that scale with a model's active parameters; they compare models with each other and are
not measurements.

| Model | Price per 1M tokens (in / out) | Energy estimate (Wh per 1k tokens) |
|---|---|---|
| `gemma-3-4b` | $0.04 / $0.05 | 0.05 |
| `gpt-oss-20b` | $0.018 / $0.09 | 0.06 |
| `gemma-4-31b` | $0.20 / $0.40 | 0.30 |
| `gpt-oss-120b` | $0.15 / $0.60 | 0.15 |
| `deepseek-v4-pro` | $0.89 / $1.77 | 0.50 |
