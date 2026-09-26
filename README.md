# Researcher Matching Baseline

## Week 2 status

The Week 2 pipeline uses OpenAlex data and preserves the many-to-many paper-author relationship:

- `Data/raw/rawdata.csv`: 12,931 author-paper rows.
- `Data/processed/papers.csv`: 2,884 unique papers.
- `Data/processed/paper_author.csv`: 12,316 valid paper-author relationships.
- `Data/processed/researcher_profiles.csv`: 9,941 unique researcher profiles built from titles, abstracts, topics, and keywords.

Cleaning findings are retained in the data rather than silently discarded: 730 unique papers have no abstract and 1 has no publication year. Empty author IDs are excluded from paper-author links and profiles.

The raw-to-processed transformation is reproducible with `uv run python src/preprocess.py`. It removes blank IDs, normalizes whitespace, preserves papers even when abstracts are missing, and removes duplicate `(work_id, author_id)` relationships.

The baseline is TF-IDF with unigram/bigram features, sublinear term frequency, and cosine similarity. `src/baseline.py` provides interactive retrieval, while `src/create_evaluation_set.py` creates the original manually labeled candidate set.

## Reproducible train/test evaluation

`src/evaluate_baseline.py` performs a deterministic 80/20 split by unique paper (`random_state=42`). Profiles are built only from training papers and grouped by stable OpenAlex author ID. Each test paper is a held-out query, and its known training-set authors are the relevance labels. It writes:

- `Data/processed/calculated/train_papers.csv`
- `Data/processed/calculated/test_papers.csv`
- `Data/processed/calculated/heldout_queries.csv`
- `Data/processed/calculated/heldout_evaluation_candidates.csv`
- `Data/processed/calculated/heldout_metrics.csv`
- `Data/processed/calculated/established_researcher_metrics.csv`

Run the complete baseline evaluation with:

```bash
uv run python src/evaluate_baseline.py
```

This is an intrinsic validation split, not expert judgment. Papers whose authors are absent from the training corpus cannot be evaluated as known-researcher retrieval targets and are excluded from the held-out query table. The earlier manually labeled metrics should therefore be reported as exploratory validation, not definitive human-ground-truth performance.

The split contains 577 test papers, but only 303 have at least one author represented in training; 52.5% are evaluated and 47.5% are cold-start exclusions. The final overall mean metrics are Precision@5 `0.1261`, Precision@10 `0.0845`, Recall@10 `0.2791`, MRR `0.2033`, and Hit@10 `0.3267`. These metrics should be interpreted as initial baseline results, not classification accuracy: the system ranks researchers for a query and does not predict a single class.

For the intended profile-matching use case, the evaluator also reports an established-researcher slice for authors with at least two papers in the full corpus. That slice has 303 queries and mean Precision@5 `0.1835`, Precision@10 `0.1198`, Recall@10 `0.4391`, MRR `0.3297`, and Hit@10 `0.5083`; it is reported alongside the overall result because it excludes genuine one-paper cold-start researchers rather than pretending they have a usable history.

The 50-row manually labeled candidate file and the 50-row annotated-pairs file are included as exploratory annotation material. They are not used as definitive gold labels because they do not provide expert relevance judgments for the OpenAlex researcher IDs in the retrieval corpus.
