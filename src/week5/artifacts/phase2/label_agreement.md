# Old vs new labels (Week 5, Phase 2)

Old label: `existing_label` (TF-IDF-derived; blank was scored as not relevant). New label: `manual_relevance` from `judging/manual_judgments_completed.csv`; levels 2–3 count as relevant.

Label provenance: judge_id {'Codex_AIReview': 214}, 1 distinct `judged_at` value(s). These labels were produced by an AI reviewer, not a human judge.

| Slice | Rows | Match | Differ | Disagreement rate | Old relevant | New relevant |
|---|---|---|---|---|---|---|
| All pooled pairs | 214 | 86 | 128 | 59.8% | 80 | 192 |
| Explicit old labels (existing_label = 1) | 80 | 72 | 8 | 10.0% | 80 | 72 |
| Implied old negatives (existing_label blank) | 134 | 14 | 120 | 89.5% | 0 | 120 |

New level by old label:

| Old label | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 1 | 4 | 4 | 32 | 40 |
| blank | 2 | 12 | 45 | 75 |

## By query

| Query | Rows | Match | Differ | Disagreement rate | Old relevant | New relevant |
|---|---|---|---|---|---|---|
| q01 natural language processing | 30 | 17 | 13 | 43.3% | 10 | 23 |
| q02 machine learning | 33 | 14 | 19 | 57.6% | 10 | 29 |
| q03 transformer based text classification | 25 | 9 | 16 | 64.0% | 10 | 24 |
| q04 information retrieval | 26 | 8 | 18 | 69.2% | 10 | 24 |
| q05 knowledge graphs | 20 | 9 | 11 | 55.0% | 10 | 13 |
| q06 battery materials and energy storage | 27 | 10 | 17 | 63.0% | 10 | 27 |
| q07 quantum computing | 24 | 9 | 15 | 62.5% | 10 | 23 |
| q08 deep learning neural networks | 29 | 10 | 19 | 65.5% | 10 | 29 |

## By system (each pair counted for every system that returned it)

| System | Rows | Match | Differ | Disagreement rate | Old relevant | New relevant |
|---|---|---|---|---|---|---|
| Week 3 (hybrid) | 80 | 33 | 47 | 58.8% | 30 | 73 |
| Week 3 (semantic) | 80 | 22 | 58 | 72.5% | 16 | 72 |
| Week 3 (tfidf) | 80 | 72 | 8 | 10.0% | 80 | 72 |
| Week 4 (average) | 80 | 21 | 59 | 73.8% | 15 | 72 |
| Week 4 (best_paper) | 80 | 23 | 57 | 71.2% | 16 | 71 |
| Week 4 (top_k_average) | 80 | 21 | 59 | 73.8% | 15 | 72 |

## Rows flagged for review

131 rows have `review_flag = yes`. Reasons (from the evidence columns):

- no Week 4 evidence; judged from profile titles only: 127
- Week 4 evidence present but weak or non-substantive: 4

By query: q01: 19, q02: 24, q03: 15, q04: 15, q05: 10, q06: 16, q07: 13, q08: 19

Levels when flagged: {'0': 6, '1': 11, '2': 62, '3': 52}; when not flagged: {'0': 0, '1': 5, '2': 15, '3': 63}.

Weak-evidence rows:

- q02 Parthiv Sarkar (level 1): Code for the ML models || Code for the ML models || Code for the ML models
- q02 Seema Shah (level 1): Retraction Note to: A Survey on Applications of Machine Learning Algorithms in Health care
- q02 Samit Nikesh Shah (level 1): Retraction Note to: A Survey on Applications of Machine Learning Algorithms in Health care
- q07 Jeya Mala (level 1): Editorial
