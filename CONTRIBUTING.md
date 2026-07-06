# Contributing to CAFE

Thanks for your interest in CAFE (**C**ompound-**A**I **F**actorial **E**valuation) — a
design-of-experiments platform for measuring which techniques actually drive quality in a compound AI
system. Contributions of all kinds are welcome: bug reports, new example techniques, statistics or
design methods, docs, and UI work.

## Repository layout

```
packages/cafe-core/     the Python library (the engine + judge + stats). This is the pip package.
apps/web-app/           the self-hostable platform: FastAPI backend + React (Vite) frontend.
apps/landing/           the marketing / landing page (static).
techniques/             example pipelines — the extension point users copy and adapt.
examples/               tutorial notebooks (rendered into the docs).
docs/                   MkDocs documentation source.
```

## Development setup

CAFE needs **Python ≥ 3.11**. Install the library in editable mode with the stats + dev extras:

```bash
git clone https://github.com/fabian-lu/Cafe.git
cd Cafe
python -m venv .venv && source .venv/bin/activate
pip install -e "packages/cafe-core[stats,dev]"
```

The core engine has no hard dependencies; the `stats` extra pulls in pandas/numpy/statsmodels/scipy/
matplotlib for the analysis layer. The ordinal CLMM and binary GLMM models additionally need **R** with
the `ordinal` / `lme4` packages — CAFE degrades gracefully without them. Check your environment with:

```bash
cafe doctor
```

### Tests & linting

```bash
cd packages/cafe-core
pytest                 # unit tests
ruff check .           # lint
ruff format .          # format
```

Please add or update tests for any behavior change, and keep `ruff check` clean.

### Running the web app

```bash
cd apps/web-app
cp .env.example .env   # add your LLM keys (never commit .env)
docker compose up
```

Frontend at `http://localhost:5173`, backend at `http://localhost:8000`.

## Adding a technique / pipeline

Each file in `techniques/` that exposes a `cafe.Pipeline` with a `@pipe.compose` is a self-contained
**system under test**. To add your own, copy `techniques/pipeline.py`, register your stages/techniques
with `@pipe.technique(...)`, and compose them. See the docs "Define a system" guide. The registry is the
intended extension point — you should not need to modify `cafe-core` to plug in a new system.

## Pull requests

1. Fork and branch off `main` (`git checkout -b my-change`).
2. Keep changes focused; one logical change per PR.
3. Ensure `pytest` passes and `ruff check` is clean.
4. Write a clear PR description: what changed and why.
5. For new features, update the docs and (if user-facing) the relevant example notebook.

By contributing you agree that your contributions are licensed under the repository's LICENSE.

## Reporting bugs

Open an issue with: what you ran, what you expected, what happened (include the traceback), and your
environment (`cafe doctor` output helps). Minimal reproductions are hugely appreciated.

## Citation

If you use CAFE in your research, please cite the paper (see the README for the BibTeX entry / arXiv
link once available).
