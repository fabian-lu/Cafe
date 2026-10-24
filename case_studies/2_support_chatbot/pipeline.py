"""The system under test: a customer-support chatbot.

    customer message → system_prompt → model → reply

Each stage has several interchangeable techniques. CAFE turns every stage into a factor and
tests all combinations:

- system_prompt: plain · with_support_handbook
- model:         gemma-3-4b · gpt-oss-20b · gpt-oss-120b · deepseek-v4-pro

The web app runs this same file, so the notebook and the app evaluate the identical system.
Needs OLLAMA_API_KEY and OPENROUTER_API_KEY in the repo's .env.
"""

import cafe

# ── Prices ───────────────────────────────────────────────────────────────────────────────────────
# Ollama Cloud doesn't report what a call costs, so we tell CAFE each model's market price
# (OpenRouter list price, October 2026). CAFE expects USD per 1,000 tokens, hence "per million / 1000".
# From then on, CAFE prices every call made through cafe.complete() with this model.
cafe.set_model_cost(model="openrouter/google/gemma-3-4b-it", per_1k_input=0.04 / 1000, per_1k_output=0.05 / 1000)
cafe.set_model_cost(model="ollama_cloud/gpt-oss:20b", per_1k_input=0.018 / 1000, per_1k_output=0.09 / 1000)
cafe.set_model_cost(model="ollama_cloud/gpt-oss:120b", per_1k_input=0.15 / 1000, per_1k_output=0.60 / 1000)
cafe.set_model_cost(model="ollama_cloud/deepseek-v4-pro", per_1k_input=0.89 / 1000, per_1k_output=1.77 / 1000)


# ── The two system prompts ───────────────────────────────────────────────────────────────────────
# Both share a length limit, so the comparison is about substance and not verbosity (LLM judges
# tend to reward longer answers), and a note on the {{...}} placeholders in the Bitext data.
# The ONLY difference between the two is the support handbook: the company's own procedures, the
# way a real support team would put them into its bot's prompt.
LENGTH_LIMIT = "Keep your reply under 300 characters."
PLACEHOLDER_NOTE = (
    "The customer's message may contain redacted placeholders in double braces (e.g. "
    "{{Order Number}}, {{Invoice Number}}, {{Customer Support Phone Number}}) standing in for the "
    "customer's real details or company info, hidden for privacy. Treat them as if they hold real "
    "values, and you may reuse the same {{...}} tokens in your reply."
)
# The handbook was written from Bitext reference replies OUTSIDE the 100 study tickets: it describes
# how things work at the (fictional) company, not the answers to the test tickets.
SUPPORT_HANDBOOK = """SUPPORT HANDBOOK: how things work at our company.

Tone: friendly and professional; acknowledge the customer's concern. If you need details the customer
hasn't given (order number, invoice number, email address), ask for them before giving steps. If
something can't be solved in the chat, give the customer service contact.

Account
- Create an account: ask for full name, email address and preferred username, then guide them through
  the sign-up on {{Website URL}}.
- Edit, switch or delete an account: log in -> account settings / "Profile" -> the option to edit the
  details, to switch to the {{Account Type}} account ("Upgrade" / "Switch Account Type"), or to delete
  or close the account. Deleting removes all data permanently.
- Forgotten password: {{Login Page URL}} -> "{{Forgot Password}}" -> enter the account's email address
  -> follow the link in the email (check the spam folder).
- Registration problems: ask what exactly goes wrong (error message, which step) before suggesting a fix.

Orders
- Place or change an order: ask which items, or which change (items, quantity, shipping method).
- Cancel an order: log in -> "{{Online Order Interaction}}" -> the order {{Order Number}} -> "Cancel".
  If the option is missing, the order may already have shipped: refer to customer service.
- Cancellation fees: listed in the account under "Terms and Conditions" / "Service Agreement".
- Track an order or ask about the delivery date: ask for the {{Order Number}} or {{Tracking Number}}
  and check the status.

Delivery and shipping
- Delivery options: shown in the "Shipping" / "Delivery" section, with delivery times and costs. We
  also ship internationally, e.g. to {{Delivery Country}}.
- Add or change a shipping address: log in -> "My Account" / "Profile" -> "Shipping Addresses" -> add
  or edit -> save, and double-check the address.

Payment, invoices and refunds
- Payment methods: credit and debit cards (Visa, Mastercard, American Express), PayPal, bank transfer,
  Apple Pay and Google Wallet.
- Payment problems: ask what happened (error message, payment method) before suggesting a fix.
- Find or download an invoice: log in -> "Billing" / "Invoices" ("Billing History") -> invoice
  {{Invoice Number}} -> view or download it as a PDF.
- Refund policy: refunds depend on the conditions in our terms (e.g. the product is returned in its
  original condition, or within the return period); the terms are on {{Website URL}}.
- Request a refund: ask what the refund is for (product or service, amount, order details) before
  explaining the next steps.
- Refund status: acknowledge the amount ({{Currency Symbol}}{{Refund Amount}}) and say you are checking
  the status.

Contact and feedback
- Customer service: during {{Customer Support Hours}} at {{Customer Support Phone Number}}, by live chat
  on {{Website URL}}, or by email.
- Talk to a human: ask briefly what it is about, so the right person can take over.
- Complaints: take them seriously; ask for the details (what happened, order number, any documents).
- Reviews: on the product page or in the review section of {{Website URL}}.
- Newsletter: ask for the email address to check the subscription; if newsletters don't arrive, check
  the spam folder."""


# A CAFE pipeline: a set of stages, each with interchangeable techniques.
pipe = cafe.Pipeline()

# @pipe.technique(stage=..., name=...) registers the function below it as one technique of a stage.
# All techniques of a stage become the levels of that stage's factor. The optional
# energy_wh_per_1k_tokens is the estimated energy per 1,000 tokens (see the README); CAFE multiplies
# it by the tokens of every cafe.complete() call made inside the technique.


# ── Stage 1: the system prompt ───────────────────────────────────────────────────────────────────
@pipe.technique(stage="system_prompt", name="plain", description="A bare support assistant, without the handbook.")
async def plain_prompt(ctx):
    return f"You are a customer-support assistant. {LENGTH_LIMIT} {PLACEHOLDER_NOTE}"


@pipe.technique(stage="system_prompt", name="with_support_handbook",
                description="The same assistant plus the company's support handbook (its procedures).")
async def prompt_with_support_handbook(ctx):
    return (f"You are a customer-support assistant. Follow the support handbook below. {LENGTH_LIMIT} "
            f"{PLACEHOLDER_NOTE}\n\n{SUPPORT_HANDBOOK}")


# ── Stage 2: the model that writes the reply (one technique per model) ───────────────────────────
@pipe.technique(stage="model", name="gemma-3-4b", energy_wh_per_1k_tokens=0.05,
                description="Gemma 3 4B (Google, 4B parameters) writes the reply.")
async def gemma_3_4b(ctx, customer_message, system_prompt):
    return await cafe.complete(
        model="openrouter/google/gemma-3-4b-it",   # retired on Ollama Cloud, so via OpenRouter
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": customer_message}],
        temperature=0.3,
        num_retries=3,   # retry if the provider is briefly rate-limited
    )


@pipe.technique(stage="model", name="gpt-oss-20b", energy_wh_per_1k_tokens=0.06,
                description="GPT-OSS 20B (OpenAI open-weight, mixture of experts) writes the reply.")
async def gpt_oss_20b(ctx, customer_message, system_prompt):
    return await cafe.complete(
        model="ollama_cloud/gpt-oss:20b",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": customer_message}],
        temperature=0.3,
        num_retries=3,
    )


@pipe.technique(stage="model", name="gpt-oss-120b", energy_wh_per_1k_tokens=0.15,
                description="GPT-OSS 120B (OpenAI open-weight, mixture of experts) writes the reply.")
async def gpt_oss_120b(ctx, customer_message, system_prompt):
    return await cafe.complete(
        model="ollama_cloud/gpt-oss:120b",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": customer_message}],
        temperature=0.3,
        num_retries=3,
    )


@pipe.technique(stage="model", name="deepseek-v4-pro", energy_wh_per_1k_tokens=0.50,
                description="DeepSeek-V4-Pro (a large mixture-of-experts model) writes the reply.")
async def deepseek_v4_pro(ctx, customer_message, system_prompt):
    return await cafe.complete(
        model="ollama_cloud/deepseek-v4-pro",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": customer_message}],
        temperature=0.3,
        num_retries=3,
    )


# ── How the stages connect ───────────────────────────────────────────────────────────────────────
# @pipe.compose marks the function that wires the stages together. CAFE calls it once for every
# configuration and ticket; ctx.run(stage=...) runs the technique the configuration chose for
# that stage (and records its time, tokens, cost and energy).
# CAFE always passes (config, item, ctx). config isn't needed here, because ctx.run reads it
# itself; you would use it for your own control flow, e.g. `if config["model"] == ...`.
@pipe.compose
async def support_chatbot(config, item, ctx):
    system_prompt = await ctx.run(stage="system_prompt")
    return await ctx.run(stage="model", customer_message=item["text"], system_prompt=system_prompt)
