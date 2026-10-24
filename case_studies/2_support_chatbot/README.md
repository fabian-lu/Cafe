# Case study 2 · A customer-support chatbot

**The question a support team asks:** do we need an expensive model for our support bot, or is a
smaller one just as good? And does it help to give the bot our own procedures (a support handbook in
the system prompt), perhaps enough for a small model to catch up with a large one?

## Design

| | |
|---|---|
| Tickets | 100 tickets from the Bitext customer-support dataset (`data/support_tickets.json`) |
| Factor `system_prompt` | `plain` · `with_support_handbook` |
| Factor `model` | `gemma-3-4b` · `gpt-oss-20b` · `gpt-oss-120b` · `deepseek-v4-pro` |
| Size | 2 × 4 = 8 configurations × 100 tickets = 800 replies |
| Rubric 1 | `support_quality`: 1 poor · 2 weak · 3 ok · 4 good · 5 excellent (ordinal) |
| Rubric 2 | `professional_tone`: 1 inappropriate · 2 acceptable · 3 professional (ordinal) |
| Judge | Claude Opus 5 (via OpenRouter), a different model family from all four chatbot models |
| Filters | `category` (11, e.g. `refund`, `order`, `payment`) · `intent` (e.g. `track_refund`) |

Both system prompts share a 300-character length limit, so the comparison is about substance, not
length (LLM judges tend to reward longer answers). The only difference is the **support handbook**:
the company's own procedures (how to cancel an order, where invoices are, how refunds work, how to
reach customer service), the way a real support team would put them into its bot's prompt. The
handbook was written from Bitext reference replies *outside* the 100 study tickets, so it describes
how things work at the (fictional) company without containing the answers to the test tickets.
It adds about 800 tokens to every prompt, so the Pareto chart also shows whether it pays for itself.

## Files

| File | What it is |
|---|---|
| `pipeline.py` | The system under test: system prompt → model. The web app runs the same file. |
| `support_chatbot_study.ipynb` | The study: data, factors, rubrics, judge, run, second rubric, report, save. |
| `data/prepare_data.py` | Downloads Bitext and saves the fixed sample of 100 tickets (already done). |
| `data/support_tickets.json` | The 100 tickets (see `data/SOURCE.md`). |
| `results/support_chatbot_results.json` | The saved results with both rubrics; import them in the web app. |

## How to run

From the repository root, with API keys for Ollama Cloud and OpenRouter in `.env`:

```bash
pip install -e "packages/cafe-core[all]"
cd case_studies/2_support_chatbot
jupyter lab support_chatbot_study.ipynb
```

The run saves its progress as it goes, so an interrupted run continues where it stopped. Most of
the cost is the judge (Claude Opus 5, about 1,600 judge calls).

## Cost and energy figures

As in case study 1: every model is charged at its OpenRouter list price (October 2026), set at the
top of `pipeline.py`, and the energy figures are estimates per 1,000 tokens that scale with a
model's active parameters. DeepSeek-V4-Pro is a large mixture-of-experts model served across several
GPUs, so it gets the highest figure. The figures compare models with each other; they are not
measurements.

## Results

| Support quality (1–5) | plain | with handbook | cost per 1,000 replies (with handbook) |
|---|---|---|---|
| Gemma 3 4B | 2.96 | 3.51 | $0.04 |
| GPT-OSS 20B | 3.70 | 4.20 | $0.03 |
| GPT-OSS 120B | 3.99 | **4.48** | $0.22 |
| DeepSeek-V4-Pro | 4.09 | 4.19 | $0.99 |

- **The handbook helps** (3.69 → 4.09, d = 0.39): about half a point for Gemma 3 4B and both GPT-OSS
  models, but only 0.1 for DeepSeek-V4-Pro (interaction coefficients p ≈ .02; joint test p = .05).
- **The most expensive model isn't needed.** GPT-OSS 120B with the handbook is the best configuration
  at under a quarter of DeepSeek's cost; GPT-OSS 20B with the handbook equals DeepSeek with the
  handbook at about a thirtieth. Overall, GPT-OSS 120B and DeepSeek are statistically
  indistinguishable (d = 0.10, 95% CI [−0.09, 0.30]); Gemma 3 4B is clearly behind.
- **Knowledge beats size:** GPT-OSS 20B with the handbook (4.20) beats GPT-OSS 120B without it (3.99).
- **Tone is a different story:** the handbook improves tone for every model; GPT-OSS 120B is the
  friendliest, DeepSeek accurate but curt.
- The judge was the expensive part: about $16 for the 1,600 Claude Opus judgements, against under
  $0.10 for all 800 replies.
