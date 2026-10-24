# Data source

**`support_tickets.json`**: 100 tickets drawn at random (seed 0) from the Bitext customer-support
dataset, https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset,
created by `prepare_data.py`.

Fields: `id`, `customer_message`, `reference_reply` (Bitext's reply, used by the judge as a guide to
the correct information), `category` and `intent`.

The tickets are **synthetic**: Bitext describes the dataset as "hybrid synthetic", generated from
natural seed texts with their language-generation tools and curated by linguists. They are
realistic support requests, but not messages from real customers. Personal details appear as
placeholders such as `{{Order Number}}`.

License: Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0).
