"""The system under test: an invoice-processing pipeline.

    invoice PDF → text → extract the booking data → (optionally) check it → booking data as JSON

It is written like a company's existing system: plain Python, nothing from CAFE inside. CAFE
only calls `process_invoice(config, invoice)` and reads what it returns (black-box mode). The
config says which model to use, how to turn the PDF into text, and whether to add a checking step:

- model:     gemma-3-4b · gpt-oss-20b · gemma-4-31b · gpt-oss-120b · deepseek-v4-pro
- pdf_text:  with_layout · without_layout
- prompting: extract_only · extract_then_check

CAFE can't look inside a black box, so it can't count tokens or price the calls. That's why the
system keeps its own price and energy table and reports cost, tokens and energy itself.
Needs OLLAMA_API_KEY and OPENROUTER_API_KEY in the environment.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import litellm
from pypdf import PdfReader

INVOICE_FOLDER = Path(__file__).parent / "data" / "invoices"
litellm.suppress_debug_info = True   # no "Provider List" line printed for every call


# ── The models, with their price and estimated energy ────────────────────────────────────────────
@dataclass
class Model:
    provider: str              # "ollama_cloud" or "openrouter"
    name: str                  # the model's name at that provider
    usd_per_1m_input: float    # market price per million input tokens (OpenRouter list price, Oct 2026)
    usd_per_1m_output: float   # market price per million output tokens
    wh_per_1k_tokens: float    # estimated energy per 1,000 tokens (see the README)


MODELS = {
    "gemma-3-4b": Model(provider="openrouter", name="google/gemma-3-4b-it",   # retired on Ollama Cloud
                        usd_per_1m_input=0.04, usd_per_1m_output=0.05, wh_per_1k_tokens=0.05),
    "gpt-oss-20b": Model(provider="ollama_cloud", name="gpt-oss:20b",
                         usd_per_1m_input=0.018, usd_per_1m_output=0.09, wh_per_1k_tokens=0.06),
    "gemma-4-31b": Model(provider="ollama_cloud", name="gemma4:31b",
                         usd_per_1m_input=0.20, usd_per_1m_output=0.40, wh_per_1k_tokens=0.30),
    "gpt-oss-120b": Model(provider="ollama_cloud", name="gpt-oss:120b",
                          usd_per_1m_input=0.15, usd_per_1m_output=0.60, wh_per_1k_tokens=0.15),
    "deepseek-v4-pro": Model(provider="ollama_cloud", name="deepseek-v4-pro",
                             usd_per_1m_input=0.89, usd_per_1m_output=1.77, wh_per_1k_tokens=0.50),
}


# ── The prompts ──────────────────────────────────────────────────────────────────────────────────
FIELD_DESCRIPTIONS = """- vendor_name: the company that issued the invoice
- document_number: the invoice or credit-note number as printed (null if none is printed)
- invoice_date: the invoice date, as YYYY-MM-DD
- due_date: the payment due date, as YYYY-MM-DD (null if none is stated)
- currency: the ISO 4217 code of the amount payable (e.g. EUR, USD, GBP, ZAR)
- tax: the total tax amount (0 if there is none)
- total: the total amount payable
- document_type: "invoice", "credit_note", or "other" (e.g. a statement or a quote)

Give all amounts in the currency of the amount payable, as plain numbers without currency symbols
or thousands separators."""

EXTRACT_PROMPT = """Extract the booking data from the invoice below and return it as a JSON object with these keys:

{fields}

Return only the JSON object.

INVOICE:
{invoice_text}"""

CHECK_PROMPT = """Below are an invoice and booking data that were extracted from it. Check every value against
the invoice and correct any mistake. In particular: is the total really the amount payable, and is the
currency the currency of that amount? Do tax and total add up with the amounts on the invoice? Is the
document number the invoice or credit-note number (not an order or reference number)?

The keys mean:
{fields}

Return only the corrected JSON object, with the same keys.

INVOICE:
{invoice_text}

EXTRACTED DATA:
{extracted_data}"""


# ── Two helpers: reading a PDF, and calling a model ──────────────────────────────────────────────
def pdf_to_text(pdf_path, keep_layout):
    """Turn an invoice PDF into text, either keeping the page layout or as plain reading order.

    "plain" is pypdf's default: whoever calls extract_text() without thinking about it gets it.
    """
    mode = "layout" if keep_layout else "plain"
    return "\n".join(page.extract_text(extraction_mode=mode) for page in PdfReader(pdf_path).pages)


async def ask(model, prompt):
    """Send one prompt to a model (via LiteLLM); return its reply and the input and output tokens."""
    if model.provider == "ollama_cloud":   # Ollama Cloud speaks the OpenAI protocol
        connection = {"model": f"openai/{model.name}", "api_base": "https://ollama.com/v1",
                      "api_key": os.environ["OLLAMA_API_KEY"]}
    else:
        connection = {"model": f"openrouter/{model.name}"}
    response = await litellm.acompletion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=180,
        num_retries=3,   # retry if the provider is briefly rate-limited
        **connection,
    )
    reply = response.choices[0].message.content or ""
    return reply, response.usage.prompt_tokens, response.usage.completion_tokens


# ── The whole system, as CAFE calls it ───────────────────────────────────────────────────────────
async def process_invoice(config, invoice):
    model = MODELS[config["model"]]

    # Step 1: PDF → text, with or without the layout.
    invoice_text = pdf_to_text(pdf_path=INVOICE_FOLDER / f"{invoice['id']}.pdf",
                               keep_layout=(config["pdf_text"] == "with_layout"))

    # Step 2: extract the booking data.
    prompt = EXTRACT_PROMPT.format(fields=FIELD_DESCRIPTIONS, invoice_text=invoice_text)
    reply, input_tokens, output_tokens = await ask(model=model, prompt=prompt)

    # Step 3 (optional): a second call checks the extracted data against the invoice and corrects it.
    if config["prompting"] == "extract_then_check":
        prompt = CHECK_PROMPT.format(fields=FIELD_DESCRIPTIONS, invoice_text=invoice_text, extracted_data=reply)
        reply, more_input_tokens, more_output_tokens = await ask(model=model, prompt=prompt)
        input_tokens += more_input_tokens
        output_tokens += more_output_tokens

    # Step 4: the JSON object in the reply (models sometimes wrap it in ``` fences or add a sentence).
    match = re.search(r"\{.*\}", reply, flags=re.DOTALL)
    try:
        extracted_fields = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        extracted_fields = None

    # What CAFE reads: the answer, plus the cost, tokens and energy this system reports about itself.
    tokens = input_tokens + output_tokens
    return {
        "output": json.dumps(extracted_fields, indent=1) if extracted_fields is not None else reply,
        "cost_usd": (input_tokens * model.usd_per_1m_input + output_tokens * model.usd_per_1m_output) / 1_000_000,
        "tokens": tokens,
        "energy_wh": tokens / 1000 * model.wh_per_1k_tokens,
    }
