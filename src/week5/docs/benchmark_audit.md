# Benchmark audit (Week 5, Phase 1)

This audit describes the benchmark used by every Week 3 and Week 4 comparison
so far. All figures come from the frozen runs `20261007T154203Z` and
`20261007T154330Z` in `src/week5/artifacts/runs/`.

## Where the labels come from

- File: `Data/processed/calculated/evaluation_candidates_labeled.csv`, written by
  `src/week2/create_evaluation_set.py`.
- 8 queries × 10 rows = 80 labelled researchers, and all 80 are marked relevant.
  There are no researchers judged non-relevant.
- For each query, the 10 labelled researchers are TF-IDF's own top 10. Labels
  were assigned by a keyword-matching rule, not by human judgment.
- Confirmed in the frozen runs: Week 3 TF-IDF returns exactly the labelled set
  for every query (Precision@10 = 1.0 on all 8).

## Why this biases the evaluation

- Any researcher that TF-IDF did not rank in its top 10 is unjudged, and the
  evaluation counts unjudged researchers as non-relevant.
- TF-IDF therefore scores 1.0 on every metric by construction. Every other
  system loses credit for each researcher it returns that TF-IDF did not, even
  if that researcher is relevant.
- The scores measure agreement with TF-IDF's candidates, not true relevance.
  They cannot show whether semantic or paper-level retrieval is better or
  worse than TF-IDF.

## Why the frozen baseline is still useful

- It is the comparison every earlier week reported, so it serves as a fixed
  historical reference.
- It is fully reproducible: two independent runs on the same index produced
  byte-identical metrics, rankings and evidence.
- The comparison between non-TF-IDF systems (Week 3 semantic and hybrid,
  Week 4 strategies) remains informative about how much each one agrees with
  keyword retrieval. It still says nothing about true relevance.

## Measured results on this benchmark

| System | P@5 | P@10 | Recall@10 | MRR | nDCG@10 | Hit@10 |
|---|---|---|---|---|---|---|
| Week 3 semantic | 0.300 | 0.200 | 0.200 | 0.641 | 0.261 | 0.750 |
| Week 3 TF-IDF | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Week 3 hybrid | 0.525 | 0.375 | 0.375 | 0.875 | 0.471 | 0.875 |
| Week 4 best_paper | 0.250 | 0.200 | 0.200 | 0.348 | 0.196 | 0.750 |
| Week 4 average | 0.275 | 0.188 | 0.188 | 0.348 | 0.189 | 0.750 |
| Week 4 top_k_average | 0.275 | 0.188 | 0.188 | 0.348 | 0.189 | 0.750 |

## Next step (later phase, not done here)

A fair benchmark will pool the top 10 of all six systems per query and have
the pooled researchers judged manually. Phase 1 only exports the pool:
`candidate_pool.csv` in each run folder lists 214 unique query–researcher
pairs (20–33 per query). Only 80 of them carry an existing label; the other 134
are unjudged. Its `manual_relevance` column is deliberately empty. No relevance
labels have been generated automatically.
