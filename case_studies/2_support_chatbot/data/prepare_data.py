"""Download the Bitext customer-support dataset and save a fixed sample of 100 tickets.

Run once from this folder (`python prepare_data.py`); the study notebook then reads
`support_tickets.json`, so it works offline and always uses the same tickets.
"""

import json

from datasets import load_dataset

NUMBER_OF_TICKETS = 100
SEED = 0   # the same sample as the earlier version of this study

bitext = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
sample = bitext.shuffle(seed=SEED).select(range(NUMBER_OF_TICKETS))

tickets = [
    {
        "id": f"ticket-{number:03d}",
        "customer_message": row["instruction"],
        "reference_reply": row["response"],
        "category": row["category"].lower(),   # e.g. refund, shipping, payment
        "intent": row["intent"].lower(),       # e.g. track_refund, cancel_order
    }
    for number, row in enumerate(sample, start=1)
]

with open("support_tickets.json", "w") as file:
    json.dump(tickets, file, indent=1, ensure_ascii=False)

print(f"saved {len(tickets)} tickets from {len({t['category'] for t in tickets})} categories")
