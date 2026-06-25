worth noting:
- IMPORTANT: Juding works with and without references!!!

Problems:
- how does providing the system even work? is it the users responsibility to do the full factorial sweep?
- on github we have to show how to cite our system/paper
- a lot of the time, im sure users want to test if one specific thign from compound ai does substantially help with the asnwers.. e.g. : if i add web search capabilites to my system, does it make the thing better? or maybe two thigns, e.g. web search + pdf/RAG .. can we compare systems with each other?
- logo



Proposed sequence

  ┌─────────────────────┬───────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────┐
  │        Slice        │                                What ships                                 │                         Proves / why first                          │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 0. Scaffold         │ git init, monorepo skeleton, pyproject, lint                              │ It's not a repo yet; CLAUDE.md says set it up at port start. Small. │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │                     │ Study/Factor (pure, no DB) + full-factorial generation + async resumable  │ The core reframe. Black box + declared factors, crash-safe          │
  │ 1. cafe-core engine │ executor + a neutral example system + cafe run headless → returns results │ execution, library-first — all the things that diverge from DIVA.   │
  │                     │  object                                                                   │ No web, no DB.                                                      │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 2. Judge + stats    │ Port the LLM judge + the 3-layer stats (DIVA's stats.py is good — reuse   │ Now cafe run gives attribution + significance + CLMM. The paper's   │
  │                     │ it) into core                                                             │ spine, headless.                                                    │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 3. Designs +        │ Fractional factorial, Pareto, then Optimize mode (BO/bandits — the risky  │ Rounds out the methodology; Optimize is isolated so it's cuttable.  │
  │ analysis            │ one)                                                                      │                                                                     │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 4. Web backend      │ FastAPI + worker reusing the same engine + Postgres + API                 │ The platform wraps core; no logic duplicated.                       │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 5. Frontend         │ Factorial Mono UI                                                         │ The beautiful, non-negotiable surface.                              │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 6. Landing + docs + │ Static read-only demo, docs site, contribution guide                      │ The demo-track deliverables.                                        │
  │  demo snapshot      │                                                                           │                                                                     │
  └─────────────────────┴───────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────┘
