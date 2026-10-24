"""Download the Invoice Extraction Benchmark and save the invoices this study uses.

Run once from this folder (`python prepare_data.py`). It keeps the 106 invoices that are digital
PDFs (with a text layer) and leaves out:

- the 68 scanned images (they would need OCR, a different pipeline), and
- 7 documents built to test something else: files holding several invoices, and invoices with
  deliberately wrong totals.

It writes `invoices/<id>.pdf` and `correct_fields.json` (the correct booking data per invoice).
"""

import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

# The fields the system has to extract, mapped to the benchmark's names for them.
FIELDS = {
    "vendor_name": "vendor_name",
    "document_number": "invoice_number",
    "invoice_date": "invoice_date",
    "due_date": "due_date",
    "currency": "currency",
    "tax": "tax",
    "total": "total",
    "document_type": "document_type",
}

download = Path(snapshot_download(
    repo_id="drew-ipp/invoice-extraction-benchmark",
    repo_type="dataset",
    allow_patterns=["corpus/v1/metadata.jsonl", "corpus/v1/documents/*.pdf"],
))
corpus = download / "corpus" / "v1"
documents = [json.loads(line) for line in open(corpus / "metadata.jsonl")]

Path("invoices").mkdir(exist_ok=True)
invoices = []
for document in documents:
    answer = document["expected"]
    if document["render"] != "pdf" or answer.get("should_flag"):
        continue

    shutil.copy(corpus / document["file_name"], Path("invoices") / f"{document['doc_id']}.pdf")
    checked = answer["scored"]   # the benchmark only scores fields that are printed on the invoice
    invoices.append({
        "id": document["doc_id"],
        "category": document["category"].replace("credit", "credit_note"),
        "difficulty": document["difficulty"],
        "correct_fields": {ours: answer[theirs] for ours, theirs in FIELDS.items() if theirs in checked},
    })

with open("correct_fields.json", "w") as file:
    json.dump(invoices, file, indent=1)

print(f"saved {len(invoices)} invoices")
