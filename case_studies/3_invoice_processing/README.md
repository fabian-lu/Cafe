# Case study 3 · Invoice processing

**The question a finance team asks:** do we need a large model to extract booking data from our
invoices, or does a small one do the job? And do the other steps of the pipeline matter: how the
PDF is turned into text, and whether a second step checks the result?

## Design

| | |
|---|---|
| Invoices | 106 invoices from the Invoice Extraction Benchmark (`data/invoices/`) |
| Factor `model` | `gemma-3-4b` · `gpt-oss-20b` · `gemma-4-31b` · `gpt-oss-120b` · `deepseek-v4-pro` |
| Factor `pdf_text` | `with_layout` (columns stay aligned) · `without_layout` (plain reading order, pypdf's default) |
| Factor `prompting` | `extract_only` · `extract_then_check` (a second call checks and corrects the result) |
| Size | 5 × 2 × 2 = 20 configurations × 106 invoices = 2,120 extractions |
| Extracted fields | `vendor_name`, `document_number`, `invoice_date`, `due_date`, `currency`, `tax`, `total`, `document_type` |
| Rubric | `ready_to_book`: 1 if every field is correct, 0 if someone has to correct it (binary) |
| Judge | `FieldCheckJudge`: plain code that compares every field with the correct value |
| Filters | `category`: `clean` · `currency` · `tax` · `credit_note` · `freight` · `multipage` · `hard`, and `difficulty`: `easy` · `medium` · `hard` |

## Black-box mode

Unlike case studies 1 and 2, the system here is ordinary Python with nothing from CAFE inside, the
way an existing company system would be. CAFE only calls `process_invoice(config, invoice)` and
reads what it returns: the extracted data, plus the cost, tokens and energy the system reports
about itself (CAFE can't look inside a black box to count them).

## A judge without an LLM

When there is exactly one right answer, a code check is cheaper and more reliable than an LLM judge.
`invoice_judge.py` compares each field with the correct value and tolerates differences that don't
matter: date formats (`06/02/2026` = `2026-02-06`), number formats (`1,966.16` = `1966.16`), case,
punctuation and legal forms in company names (`Acme Supply Co.` = `ACME SUPPLY`), and currency
symbols (`€` = `EUR`). It accepts all 106 correct answers, also when they are written in these other
formats. Its reasoning lists the wrong fields, so every verdict can be traced.

## Files

| File | What it is |
|---|---|
| `invoice_system.py` | The system under test: PDF → text → extract → (optionally) check. |
| `invoice_judge.py` | The code judge. |
| `invoice_study.ipynb` | The study: data, factors, rubric, judge, run, report, save. |
| `data/prepare_data.py` | Downloads the benchmark and keeps the 106 invoices this study uses (already done). |
| `data/invoices/`, `data/correct_fields.json` | The invoice PDFs and their correct booking data (see `data/SOURCE.md`). |
| `results/invoice_processing_results.json` | The saved results; import them in the web app. |

## How to run

From the repository root, with API keys for Ollama Cloud and OpenRouter in `.env`:

```bash
pip install -e "packages/cafe-core[all]" pypdf
cd case_studies/3_invoice_processing
jupyter lab invoice_study.ipynb
```

The run saves its progress as it goes, so an interrupted run continues where it stopped.

## Cost and energy figures

As in case studies 1 and 2: every model is charged at its OpenRouter list price (October 2026) and
the energy figures are estimates per 1,000 tokens that scale with a model's active parameters
(Gemma 4 31B is a dense 31 B model, so it uses more energy per token than the mixture-of-experts
GPT-OSS models). They compare models with each other; they are not measurements.

## Results

| Model | ready to book | cost, all 424 runs | energy, all 424 runs |
|---|---|---|---|
| Gemma 3 4B | 0.66 | $0.03 | 33 Wh |
| GPT-OSS 20B | 0.96 | $0.04 | 48 Wh |
| Gemma 4 31B | 0.97 | $0.15 | 196 Wh |
| GPT-OSS 120B | 0.97 | $0.23 | 112 Wh |
| DeepSeek-V4-Pro | 0.98 | $0.99 | 395 Wh |

- **Only the model matters.** Every model from 20B up books 96–98% of the invoices correctly and is
  statistically indistinguishable from DeepSeek-V4-Pro; Gemma 3 4B fails on a third.
- **GPT-OSS 20B does it for 4% of DeepSeek's cost and an eighth of its energy.**
- **The PDF conversion and the checking step made no significant difference** (p = .19 and p = .48).
  The checking step doubles cost and time for nothing measurable here. (A pilot on 28 invoices had
  suggested effects that did not replicate.)
- **Which invoice it is matters most** (latent ICC .83): failures concentrate on invoices that are
  payable in another currency than their lines, and on two withholding-tax invoices.
- **The two withholding-tax invoices are a convention issue, not a model error.** Our prompt asks for
  the total amount payable; every large model returned the *Net Amount Due*, as asked, while the
  benchmark's answer key expects the total before withholding. This costs every model the same 2 of
  106 invoices.
