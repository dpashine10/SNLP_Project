> **Start here:** [`WEEK5_EVALUATION_REPORT.md`](WEEK5_EVALUATION_REPORT.md) is the
> Week 5 evaluation report (metrics, error cases, robustness, conclusion).

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

Both were done later, in Phase 2. The final report is
`WEEK5_EVALUATION_REPORT.md`.

# Week 5, Phase 2: Manual Judging and Error Analysis

Phase 2 reads the frozen run `artifacts/runs/20261007T154203Z/` and never
changes it. Details: `judging/README.md` (how to judge) and
`docs/phase2_status.md` (progress and limitations).

```bash
uv run python -m src.week5.scripts.prepare_judging      # judging template + editable file (never overwrites judgments)
uv run python -m src.week5.scripts.validate_judgments   # check judging/manual_judgments.csv
uv run python -m src.week5.scripts.merge_judgments      # judgments + frozen rankings -> artifacts/phase2/judged_rankings.json
uv run python -m src.week5.scripts.fair_evaluation      # metrics on judged queries only, or a "waiting" status
uv run python -m src.week5.scripts.compare_labels       # old existing_label vs new manual_relevance
uv run python -m src.week5.scripts.summarize_pool       # artifacts/phase2/pool_summary.{json,md}
uv run python -m src.week5.scripts.analyze_errors       # artifacts/phase2/ co-author ties, error cases, review queue
uv run python -m src.week5.scripts.evaluate_improved_variant  # 6 frozen systems + author-discount variant -> artifacts/phase2/improved_variant/<time>/
uv run python -m src.week5.scripts.robustness_check     # rephrased, ambiguous, Indian-domain queries -> artifacts/phase2/robustness/<time>/
uv run python -m unittest discover -s src/week5/tests -t .
```

The last two scripts rank queries live, so they need the Week 4 index.

| Path | Contents |
|---|---|
| `judging/manual_judgments.csv` | The file to fill in: 214 pooled query–researcher pairs. |
| `judging/templates/judging_template.csv` | Frozen reference used by the validator. |
| `evaluation/fair_evaluation.py` | Evaluation on judged pairs only; unjudged pairs are never treated as irrelevant. |
| `analysis/` | Pool summary, error analysis and the robustness check. |
| `evaluation/variant_evaluation.py` | Rankings for the improved Week 4 variant, in the frozen-run format. |
| `artifacts/phase2/` | Generated Phase 2 outputs (regenerated by the scripts above). |

Status: all 214 pairs are labelled in `judging/manual_judgments_completed.csv`
(AI-generated, `judge_id = Codex_AIReview`), which the scripts use when it
exists. Results and caveats: `docs/phase2_status.md`. Notes for the final
report: `docs/report_notes.md`. Submission report: `WEEK5_EVALUATION_REPORT.md`.
