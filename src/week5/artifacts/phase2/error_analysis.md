# Error analysis (Week 5, Phase 2)

Generated from the frozen Phase 1 rankings by `python -m src.week5.scripts.analyze_errors`.

## Week 4 co-author concentration

Flags: a best-evidence paper shared by at least 3 top-10 researchers; at least 3 identical scores; or a top 10 drawn from 5 or fewer distinct papers.

| System | Query | Distinct best papers | Largest shared-paper group | Largest identical-score group |
|---|---|---|---|---|
| Week 4 (best_paper) | q01 | 4 | 5 | 5 |
| Week 4 (best_paper) | q02 | 6 | 3 | 3 |
| Week 4 (best_paper) | q03 | 6 | 3 | 3 |
| Week 4 (best_paper) | q04 | 6 | 3 | 3 |
| Week 4 (best_paper) | q05 | 3 | 5 | 5 |
| Week 4 (best_paper) | q06 | 6 | 4 | 4 |
| Week 4 (best_paper) | q07 | 8 | 2 | 2 |
| Week 4 (best_paper) | q08 | 5 | 5 | 5 |
| Week 4 (average) | q01 | 4 | 4 | 4 |
| Week 4 (average) | q02 | 8 | 2 | 2 |
| Week 4 (average) | q03 | 6 | 3 | 3 |
| Week 4 (average) | q04 | 6 | 3 | 3 |
| Week 4 (average) | q05 | 3 | 5 | 5 |
| Week 4 (average) | q06 | 6 | 4 | 4 |
| Week 4 (average) | q07 | 9 | 2 | 2 |
| Week 4 (average) | q08 | 5 | 5 | 5 |
| Week 4 (top_k_average) | q01 | 4 | 4 | 4 |
| Week 4 (top_k_average) | q02 | 8 | 2 | 2 |
| Week 4 (top_k_average) | q03 | 6 | 3 | 3 |
| Week 4 (top_k_average) | q04 | 6 | 3 | 3 |
| Week 4 (top_k_average) | q05 | 3 | 5 | 5 |
| Week 4 (top_k_average) | q06 | 6 | 4 | 4 |
| Week 4 (top_k_average) | q07 | 9 | 2 | 2 |
| Week 4 (top_k_average) | q08 | 5 | 5 | 5 |

19 of 24 Week 4 system–query rankings trigger at least one flag.

## Confirmed error cases

25 cases are confirmed directly by the evidence (no relevance assumption): 25 × co-author tie inflation. See `error_cases.csv`.

## Review queue

68 top-5 researchers across Week 3 semantic and the Week 4 strategies; 0 still need a manual judgment before relevance-dependent errors (such as generic semantic similarity or a weak evidence paper) can be categorised. See `error_review_queue.csv`.
