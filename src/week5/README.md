# Week 5, Phase 1: Frozen Evaluation Evidence

Phase 1 freezes the current Week 3 vs Week 4 comparison before any model
changes. It reruns the existing evaluation and records everything needed to
reproduce and inspect it: configuration, metrics, full rankings with scores and
evidence, a reproducibility check and a benchmark audit. It does not change a
model, a retrieval setting, a ranking formula or a benchmark label.

## Commands

From the repository root, with the Week 4 index already built
(`python -m src.week4.build_chromadb`):

```bash
uv run python -m src.week5.scripts.freeze_evaluation   # one frozen run (~45 s)
uv run python -m src.week5.scripts.freeze_evaluation   # a second, independent run
uv run python -m src.week5.scripts.compare_runs        # compares the two latest runs
```

`compare_runs` also accepts two run IDs: `compare_runs RUN_A RUN_B`.

## Generated artifacts

Each run creates a new folder `artifacts/runs/<UTC timestamp>/`; earlier runs
are never overwritten.

| File | Contents |
|---|---|
| `metrics_summary.csv` / `.json` | Mean of each metric per system. |
| `per_query_metrics.csv` | Metrics and ranked researcher IDs per system and query. |
| `rankings_evidence.json` | Per system and query: the top 10 with rank, name, score and label, plus Week 4's supporting papers. |
| `candidate_pool.csv` | All researchers returned by any system per query, with an empty `manual_relevance` column for later judging. |
| `run_metadata.json` | Dataset counts, index fingerprint (before and after), retrieval settings, systems, benchmark, metrics, package versions, git commit, and a check against Week 4's saved results. |
| `run.log` | Week 4 and Week 5 log output for the run. |

`compare_runs` writes `artifacts/reproducibility/<run_a>_vs_<run_b>.json` and
`.md`. They report whether the index, metrics, rankings, scores and output
files are identical.

Documentation: `docs/benchmark_audit.md`.

## How Week 5 uses Weeks 3 and 4

Week 5 imports their code and modifies none of it:

- The six systems come from `src.week4.evaluation.build_recommenders`: the
  Week 3 baseline (`src/week2/pipeline.py` through Week 4's
  `baseline_adapter`) in `semantic`, `tfidf` and `hybrid` modes, and the
  Week 4 pipeline with `best_paper`, `average` and `top_k_average`.
- The metrics and their aggregation are Week 4's `METRICS` and `summarize`.
  Week 5 calls each system's `recommend()` directly only so it can keep the
  scores and evidence that Week 4's own runner discards.
- The existing ChromaDB index (`Data/week4/chroma/`) is opened read-only and is
  never rebuilt or copied.
- Each run checks its summary against `Data/week4/results/summary.csv`
  (`matches_week4_results` in `run_metadata.json`).

## Measured status

- Two independent runs (`20261007T154203Z` and `20261007T154330Z`) on the same
  index were identical: the same metrics, rankings and scores, and all five
  output files byte for byte.
- Both runs reproduce Week 4's saved results exactly.
- The index fingerprint deliberately ignores `chroma.sqlite3`'s modification
  time, which ChromaDB updates whenever the index is opened.

## Known benchmark bias

The benchmark's 80 labels are exactly TF-IDF's top 10 per query, all marked
relevant. TF-IDF therefore scores 1.0 by construction, and the other systems
are penalised for researchers TF-IDF did not return. See
`docs/benchmark_audit.md`.

## Out of scope for Phase 1

- Any change to models, embeddings, retrieval depth, ranking strategies or
  labels (Phase 2).
- Building or judging a fair benchmark. Only the candidate pool is exported.
- Report writing (Phase 3).
