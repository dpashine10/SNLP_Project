# Week 5 Phase 2 status

Phase 2 replaces the TF-IDF-derived benchmark labels with relevance labels
for all 214 pooled query–researcher pairs, evaluates the six frozen rankings
against them, and documents errors. Nothing in Weeks 2–4, the ChromaDB index,
the frozen rankings or the old labels was changed.

## Labels

- File: `src/week5/judging/manual_judgments_completed.csv`. It validates with
  0 errors, and its frozen columns are identical to `manual_judgments.csv`.
  The blank file is unchanged and still present.
- Levels: 115 × 3, 77 × 2, 16 × 1, 6 × 0. All 214 pairs are judged.
- **Provenance:** every row has `judge_id = Codex_AIReview` and the same
  `judged_at`. These are AI-generated labels, not independent human
  judgments. They should be described that way, and a human check of at least
  a sample is advisable before treating them as ground truth.
- `review_flag = yes` on 131 rows. 127 have no Week 4 evidence and were judged
  from profile titles only; 4 have weak or non-research evidence (a code
  artifact, two retraction notices, an editorial). Flagged rows carry
  provisional levels; none were changed.

## Old vs new labels (`artifacts/phase2/label_agreement.md`)

| Slice | Rows | Differ | Disagreement |
|---|---|---|---|
| Explicit old labels (`existing_label = 1`) | 80 | 8 | 10.0% |
| Implied old negatives (blank, scored as not relevant) | 134 | 120 | 89.5% |
| All pooled pairs | 214 | 128 | 59.8% |

The old benchmark's positives mostly hold up (72 of 80 judged relevant), but
120 of the 134 researchers it implicitly scored as wrong are judged relevant.
By system, the disagreement rate is 10% for TF-IDF, which produced the old
labels, and 59–74% for every other system. This quantifies the pooling bias
recorded in `benchmark_audit.md`.

## Fair evaluation (`artifacts/phase2/fair_evaluation/20261007T172845Z/`)

All 8 queries are fully judged, so no ranking contains an unjudged
researcher. Levels 2–3 count as relevant.

| System | P@5 | P@10 | Recall@10 | MRR | nDCG@10 | Hit@10 | nDCG@10 (graded) |
|---|---|---|---|---|---|---|---|
| Week 3 semantic | 0.900 | 0.900 | 0.387 | 0.875 | 0.887 | 1.0 | 0.821 |
| Week 3 TF-IDF | 0.975 | 0.900 | 0.382 | 1.000 | 0.928 | 1.0 | 0.821 |
| Week 3 hybrid | 0.900 | 0.913 | 0.401 | 1.000 | 0.921 | 1.0 | 0.858 |
| Week 4 best_paper | 0.900 | 0.888 | 0.397 | 1.000 | 0.898 | 1.0 | 0.874 |
| Week 4 average | 0.875 | 0.900 | 0.401 | 1.000 | 0.901 | 1.0 | 0.882 |
| Week 4 top_k_average | 0.875 | 0.900 | 0.401 | 1.000 | 0.901 | 1.0 | 0.882 |

The six systems are close on every metric. Week 4 `average`/`top_k_average`
have the highest graded nDCG@10, and Week 3 TF-IDF the highest P@5 and binary
nDCG@10. With 8 queries and AI-generated labels, these differences are small
and not tested for significance. Recall@10 is measured against the judged
pool, which holds about 24 relevant researchers per query, so it cannot reach
1.0 with only 10 results.

A sensitivity run with flagged rows treated as unjudged
(`fair_evaluation/20261007T172845Z-exclude-flagged/`) is **not** a fair
comparison. The flag mostly marks researchers without Week 4 evidence, so
excluding flagged rows removes 49–65 Week 3 entries but only 3–4 Week 4
entries, and favours Week 4 by construction.

## Error analysis (`artifacts/phase2/`)

- **Co-author tie inflation:** 25 cases covering 10 query–paper groups
  (`error_cases.csv`). In 21 of the 25 cases, every tied researcher is judged 3,
  so the tie does not lower accuracy there. The other 4 cases involve
  researchers judged 1:
  - three are the q01 chatbot paper ("Virtual Assistant for Appointment
    Booking"), whose co-authors fill ranks 3–7 for Week 4, one per Week 4
    strategy;
  - one is the q02 retraction notice ("Retraction Note to: A Survey on
    Applications of Machine Learning Algorithms in Health care", Week 4
    best_paper).
- **Top-5 results judged 0/1:** 5 researchers (`error_review_queue.csv`):
  - three q01 co-authors of the chatbot paper (Week 3 semantic and Week 4);
  - q02 "machine learning": "Code for the ML models", a code artifact, ranked
    #1 by Week 3 semantic and #2 by Week 4 (weak evidence paper);
  - q07 "quantum computing": "Editorial" ranked #1 by Week 3 semantic and #2 by
    Week 4 (weak evidence paper).

## Limitations

- The labels are AI-generated (one reviewer, one timestamp). There is no
  human judgment and no agreement measure.
- 131 of 214 labels are flagged as provisional.
- 8 queries; the pool contains only the six systems' top 10s.
- The co-author tie behaviour is documented, not fixed (Week 4 code
  unchanged).
