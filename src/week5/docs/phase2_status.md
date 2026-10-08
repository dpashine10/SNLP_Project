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
researcher. Levels 2–3 count as relevant. A rerun on 2026-10-08
(`fair_evaluation/20261008T050903Z/`) gives identical metrics. It also adds two
columns, `judged_results` and `precision_among_judged`, which are needed
when a system returns researchers outside the judged pool (see below).

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

## Week 4 ranking strategies

All three Week 4 strategies existed before Week 5, run end to end, and are in
the frozen run and the fair evaluation: `best_paper`, `average` and
`top_k_average` (`src/week4/ranking_strategy.py`). `average` and
`top_k_average` (k = 3) produce identical rankings on all 8 queries.

`weighted` is excluded. It is a registered placeholder that raises
`NotImplementedError`, because no weighting formula was ever specified. It is
not in `WEEK4_STRATEGIES` and is never evaluated.

## Improved variant: `top_k_average_author_discount`

**Motivation.** The error analysis found co-author tie inflation: a paper with
many authors gives the same score to every co-author, so a few papers fill a
Week 4 top 10.

**Change.** This variant is `top_k_average` with each hit's score divided by
the square root of that paper's author count. Single-author papers are
unchanged, and co-authors of one paper still share equal credit. It is a new
registered strategy; the three original strategies and `WEEK4_STRATEGIES` are
unchanged, so the frozen run still reproduces. Tests are in
`tests/test_variant.py`.

**Evaluation** (`artifacts/phase2/improved_variant/20261008T050358Z/`). The six
frozen systems define the judged pool and which queries are eligible. The
variant is scored on the same 8 queries; any researcher it returns outside the
judged pool is removed from its list (condensed lists), not counted as wrong.

| System | P@5 | P@10 | nDCG@10 | nDCG@10 (graded) | Unjudged removed | Judged in top 10 | Precision among judged |
|---|---|---|---|---|---|---|---|
| Week 4 top_k_average | 0.875 | 0.900 | 0.901 | 0.882 | 0 | 80 | 0.900 |
| Week 4 author_discount | 0.725 | 0.413 | 0.545 | 0.517 | 44 | 36 | 0.917 |

44 of the variant's 80 results were never judged. Its condensed lists hold
only 2–7 researchers per query, which mechanically lowers P@10, Recall@10 and
nDCG@10. Its standard metrics are therefore **not comparable** with the
other six systems. On the 36 results that are judged, 33 are relevant
(0.917), against 0.888–0.913 for the six frozen systems. That is a
difference of 1–2 results and not evidence of an improvement. The 44 pairs
are listed in `unjudged_variant_pairs.csv`. Judging them would make the
comparison fair; they were deliberately left blank.

What the variant does change measurably (`coauthor_concentration.csv`, per top
10, mean over 8 queries):

| System | Distinct best-evidence papers | Largest group sharing one paper | Top 10s with ≥ 3 researchers on one paper |
|---|---|---|---|
| Week 4 best_paper | 5.5 | 3.75 | 7 of 8 |
| Week 4 average / top_k_average | 5.9 | 3.5 | 6 of 8 |
| Week 4 author_discount | 10.0 | 1.0 | 0 of 8 |

Effect on the documented failures:
- The three q01 chatbot co-authors judged 1 leave the variant's top 5.
- The q02 code artifact also leaves its top 5.
- The q07 "Editorial" stays at rank 2. It is a single-author paper, so the
  discount cannot affect it; that failure needs evidence-quality filtering
  instead.
- The variant introduces a new level-1 researcher at q02 rank 3.

On the old TF-IDF-derived labels the variant scores P@10 0.113 and MRR 0.385,
against 0.188 and 0.348 for top_k_average. Those labels are biased (see
above), so these numbers are reported only for completeness.

## Robustness (`artifacts/phase2/robustness/20261008T050812Z/`)

All seven systems were run on query texts written for this check. Before
the check, the base queries were re-ranked; they reproduced the frozen
rankings exactly.

- **Rephrasings of q01, q06 and q07** (short keyword, detailed problem
  statement, abstract-style): 9 queries. Each result is compared with the
  same system's ranking for the base query and with the base query's
  judgments.
- **Standalone queries** (no labels): ambiguous "transformer" and "network";
  Indian-domain crop yield/monsoon and Hindi/Marathi speech recognition.

| System | Mean top-10 Jaccard with base | Same top 1 | Judged coverage | Precision among judged |
|---|---|---|---|---|
| Week 3 semantic | 0.200 | 1 of 9 | 0.38 | 0.94 |
| Week 3 TF-IDF | 0.042 | 0 of 9 | 0.19 | 1.00 |
| Week 3 hybrid | 0.062 | 1 of 9 | 0.31 | 1.00 |
| Week 4 best_paper | 0.109 | 0 of 9 | 0.23 | 0.95 |
| Week 4 average | 0.118 | 0 of 9 | 0.21 | 0.95 |
| Week 4 top_k_average | 0.113 | 0 of 9 | 0.21 | 0.95 |
| Week 4 author_discount | 0.193 | 0 of 9 | 0.21 | 0.95 |

- Every system is sensitive to phrasing. A rephrased query shares on
  average 4–20% of its top 10 with the base query.
- Abstract-style queries change results most (mean Jaccard 0.06 across
  systems), and short keywords least (0.16).
- TF-IDF is the least stable, as expected for exact term matching. Week 3
  semantic and the author-discount variant are the most stable.
- Most rephrased results were never judged (coverage 19–38%), so the high
  precision among judged results rests on 17–34 researchers per system. It
  shows that the judged researchers that recur are mostly relevant, not that
  the rephrased rankings are as good as the base rankings.

Standalone queries: the following describes only what the evidence titles in
`standalone_top5.csv` show. No relevance is claimed.
- "transformer": every evidence title shown refers to electrical power
  transformers, none to the neural architecture. The ambiguity is resolved to
  one sense without any signal to the user.
- In addition, best_paper, average and top_k_average put four co-authors of
  one paper in the top 4, while the variant's top 5 come from 5 different
  papers.
- "network": average and top_k_average take 4 of their top 5 from one
  cellular-network paper, and best_paper takes its top 5 from three papers.
  The variant's top 5 come from 5 papers on communication, governance and
  social networks.
- Indian-domain queries: the titles are on-topic for every system (monsoon
  rainfall and crop yield in Indian states; Hindi, Marathi and Odia speech).
  For the speech query, the variant's rank 5 is a Marathi text-sentiment
  paper, i.e. language-matched but not speech.
- Week 3 returns no paper evidence, so several Week 3 entries show no
  title.

## Limitations

- The labels are AI-generated (one reviewer, one timestamp). There is no
  human judgment and no agreement measure.
- 131 of 214 labels are flagged as provisional.
- 8 queries; the pool contains only the six systems' top 10s.
- The co-author tie behaviour is addressed by a new, separate strategy. The
  original strategies are unchanged. The variant's ranking quality is
  undetermined until its 44 unjudged results are judged.
- Robustness uses 13 hand-written queries; rephrasings reuse the base
  query's judgments, and standalone queries have none.
