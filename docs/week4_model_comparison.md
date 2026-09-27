# Week 4 — Baseline vs Improved Model Comparison

## Goal

Compare the existing Week 2 TF-IDF researcher retrieval baseline with the improved Sentence Transformer semantic retrieval model under the same evaluation conditions.

## Data

Primary corpus: `Data/processed/researcher_profiles.csv`.

Evaluation set: `Data/processed/calculated/evaluation_candidates_labeled.csv`.

The evaluation is a **fixed-candidate reranking experiment**: for each query, the same labeled candidate researchers are scored by each model and then re-ranked. It does not measure full-corpus recall of the Sentence Transformer model.

## Models

1. **TF-IDF baseline** — word unigram/bigram TF-IDF over researcher profile text followed by cosine similarity.
2. **Sentence Transformer** — `all-MiniLM-L6-v2` dense embeddings followed by cosine similarity.
3. **Hybrid** (supplementary) — 0.7 normalized semantic score + 0.3 normalized TF-IDF score.

## Metrics

- Precision@5
- Precision@10
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)
- NDCG@10

## Reproducibility

The comparison code is in `src/week4_evaluation.py`. It produces:

- `Data/processed/calculated/week4_query_metrics.csv`
- `Data/processed/calculated/week4_model_comparison.csv`
- `Data/processed/calculated/week4_rankings.csv`
- `Data/processed/calculated/week4_error_cases.csv`

The notebook `12_week4_baseline_vs_improved.ipynb` executes the comparison and displays the results. `13_week4_error_analysis.ipynb` inspects ranking changes and provides cases for qualitative analysis.
