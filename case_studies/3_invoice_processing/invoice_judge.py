"""The judge for the invoice study: plain code, no LLM.

When there is one right answer (the total is 1966.16, not "about 2000"), a code check is cheaper
and more reliable than an LLM judge. It compares every extracted field with the correct value and
tolerates differences that don't matter:

- names:     case, punctuation and legal forms are ignored ("Acme Supply Co." = "ACME SUPPLY")
- numbers:   compared as numbers, to the cent ("1,966.16" = 1966.16); credit notes by amount only
- dates:     compared as dates ("06/02/2026" = "2026-02-06", day first)
- currency:  symbols count as their code ("€" = "EUR")

An invoice counts as `ready_to_book` only if every checked field is right; the reasoning lists the
fields that were wrong, so every verdict can be traced in the web app.
"""

import json
import re
import unicodedata
from difflib import SequenceMatcher

from dateutil import parser as date_parser

import cafe

LEGAL_FORMS = {"pty", "ltd", "limited", "llc", "inc", "co", "corp", "corporation", "cc", "plc",
               "gmbh", "kg", "bv", "sarl", "sas", "the", "company", "and"}
CURRENCY_SYMBOLS = {"€": "EUR", "£": "GBP", "$": "USD", "US$": "USD", "R": "ZAR", "N$": "NAD"}
EMPTY = {"", "none", "null", "n/a", "na", "-"}


def plain_words(text):
    """Lower-case ASCII words without punctuation: 'Kransberg (Pty) Ltd.' → ['kransberg', 'pty', 'ltd']."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", text.casefold().replace("&", " and "))


def same_company(got, correct):
    a = " ".join(w for w in plain_words(got) if w not in LEGAL_FORMS)
    b = " ".join(w for w in plain_words(correct) if w not in LEGAL_FORMS)
    if not a or not b:
        return a == b
    shorter, longer = sorted((a, b), key=len)
    return (a == b or SequenceMatcher(None, a, b).ratio() >= 0.85
            or (shorter in longer and len(shorter) >= 0.6 * len(longer)))


# A label a model may copy along with the number: "Invoice No: 098052", "Credit note # CN-12"
NUMBER_LABEL = re.compile(r"^\s*(?:tax\s+)?(?:invoice|credit\s*note)?\s*(?:no\.?|number|#)?\s*[:#]?\s*", re.I)


def same_document_number(got, correct):
    def clean(number):
        number = NUMBER_LABEL.sub("", str(number or ""))
        number = re.sub(r"[\s#:]", "", number).upper()
        return "" if number.lower() in EMPTY else number
    return clean(got) == clean(correct)


def as_date(value):
    text = str(value or "").strip()
    if text.lower() in EMPTY:
        return None
    try:
        if re.match(r"\d{4}-\d{2}-\d{2}", text):                  # ISO: year-month-day
            return date_parser.isoparse(text[:10]).date()
        return date_parser.parse(text, dayfirst=True).date()     # otherwise day first, as in Europe
    except (ValueError, OverflowError):
        return None


def same_date(got, correct):
    return as_date(got) == as_date(correct)


def as_number(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = re.sub(r"[^\d.,\-]", "", str(value))
    decimal_comma = ("," in text and "." in text and text.rfind(",") > text.rfind(".")) or \
                    ("," in text and "." not in text and not re.search(r",\d{3}$", text))
    if decimal_comma:                                   # European style: 1.966,16 or 762,20
        text = text.replace(".", "").replace(",", ".")
    try:
        return float(text.replace(",", ""))             # thousands commas: 1,966.16 or 1,966
    except ValueError:
        return None


def same_amount(got, correct, is_credit_note):
    got, correct = as_number(got), as_number(correct)
    if correct == 0 and got is None:
        return True                       # "no tax" may be left empty
    if got is None or correct is None:
        return False
    if is_credit_note:
        got, correct = abs(got), abs(correct)   # credit notes: the amount counts, not its sign
    return abs(got - correct) < 0.005


def same_currency(got, correct):
    got = str(got or "").strip()
    return CURRENCY_SYMBOLS.get(got, got).upper() == str(correct).upper()


def same_document_type(got, correct):
    got = "_".join(plain_words(got))
    if "credit" in got:
        got = "credit_note"
    elif got in ("invoice", "tax_invoice"):
        got = "invoice"
    return got == correct


def field_is_right(field, got, correct, is_credit_note):
    if field == "vendor_name":
        return same_company(got, correct)
    if field == "document_number":
        return same_document_number(got, correct)
    if field in ("invoice_date", "due_date"):
        return same_date(got, correct)
    if field in ("tax", "total"):
        return same_amount(got, correct, is_credit_note)
    if field == "currency":
        return same_currency(got, correct)
    if field == "document_type":
        return same_document_type(got, correct)
    raise ValueError(f"unknown field {field!r}")


class FieldCheckJudge:
    """A CAFE judge. Anything with a `model` name and an async `score(...)` method can judge: CAFE
    calls score() once per answer and stores the returned verdict and reasoning."""

    model = "field-check (code, no LLM)"

    async def score(self, rubric, question, answer, reference=None):
        correct_fields = json.loads(reference)
        try:
            extracted_fields = json.loads(answer)
        except (json.JSONDecodeError, TypeError):
            extracted_fields = None

        if not isinstance(extracted_fields, dict):
            wrong = ["no valid JSON in the answer"]
        else:
            is_credit_note = correct_fields.get("document_type") == "credit_note"
            wrong = [
                f"{field}: expected {correct!r}, got {extracted_fields.get(field)!r}"
                for field, correct in correct_fields.items()
                if not field_is_right(field=field, got=extracted_fields.get(field), correct=correct,
                                      is_credit_note=is_credit_note)
            ]

        ready = 0 if wrong else 1
        reasoning = ("All fields correct." if not wrong else "Wrong: " + "; ".join(wrong))
        return cafe.JudgeOutput(value=ready, value_numeric=ready, reasoning=reasoning,
                                prompt="(code check, no prompt)", raw_response=reasoning)
