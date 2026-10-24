# Data source

The invoices come from the **Invoice Extraction Benchmark v1**,
https://huggingface.co/datasets/drew-ipp/invoice-extraction-benchmark (© 2026 InvoiceParser Pro),
licensed under Creative Commons Attribution 4.0 (CC BY 4.0).

The invoices are **synthetic**: realistic invoices and credit notes from the US, UK, EU, South
Africa and Namibia, with generated companies and amounts, so they contain no personal data.

`prepare_data.py` keeps the 106 documents that are digital PDFs with a text layer and leaves out
the 68 scanned images and 7 documents built to test flagging (files holding several invoices, and
invoices with deliberately wrong totals).

- **`invoices/<id>.pdf`**: the invoice PDFs, unchanged.
- **`correct_fields.json`**: per invoice its `id`, `category`, `difficulty`, and the correct values
  of the fields the benchmark checks for it (`correct_fields`). The benchmark's `invoice_number` is
  called `document_number` here, because credit notes have a credit-note number. Subtotals are not
  used: what counts as "before tax" is ambiguous on invoices with discounts or surcharges.
