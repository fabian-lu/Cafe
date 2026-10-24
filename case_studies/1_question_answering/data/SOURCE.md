# Data source

- **`hotpotqa_questions.json`**: 50 questions from the HotpotQA distractor validation set, level
  "hard" (Yang et al., 2018, *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question
  Answering*, EMNLP). Fields: `id`, `question`, `answer` (the gold answer), `question_type`
  (`bridge` or `comparison`).
- **`paragraphs.json`**: the 500 Wikipedia paragraphs that come with these 50 questions (each
  question's 10 context paragraphs, duplicates removed), used as the retrieval collection.
- **`paragraph_embeddings.npy`**: one bge-m3 embedding per paragraph, in the same order, so the
  pipeline only has to embed the question at run time.

License: HotpotQA is distributed under CC BY-SA 4.0 (https://hotpotqa.github.io).
