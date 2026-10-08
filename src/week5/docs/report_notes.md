# Week 5 notes for the final report

Ready-to-use points for the evaluation chapter, with the artifact behind each
number. Detailed tables: `phase2_status.md`. Benchmark problems:
`benchmark_audit.md`.

## 1. Week 4 is not "suddenly better"; it is now measured fairly

- The Week 3 and Week 4 rankings are frozen (Phase 1, `artifacts/runs/20261007T154203Z/`).
  Between the old and new results, **no ranking changed; only the labels did**.
- Old labels (`existing_label`): Week 4 P@10 ≈ 0.19–0.20 and MRR 0.35,
  against Week 3 semantic P@10 0.20 and MRR 0.64.
- New labels (`manual_relevance`, all 214 pooled pairs judged): every system
  scores P@10 0.89–0.91. Week 4 has MRR 1.0 and graded nDCG@10 0.874–0.882,
  against 0.875 and 0.821 for Week 3 semantic.
- The jump comes from removing the measurement error. 120 of the 134
  researchers that the old benchmark implicitly scored as wrong are judged
  relevant (`artifacts/phase2/label_agreement.md`).
- Correct statement for the report: under fair labels, Week 4 performs on par
  with the Week 3 baselines, with slightly higher graded nDCG. The
  differences are small, cover 8 queries, and were not tested for
  significance.

## 2. Why TF-IDF looked perfect under the old labels

- The old benchmark's 80 positives are exactly Week 3 TF-IDF's top 10 for
  each of the 8 queries, all marked relevant. TF-IDF therefore scored 1.0 on
  every metric by construction (`benchmark_audit.md`).
- Every other system was penalised for any researcher TF-IDF had not
  returned, relevant or not. This is pooling bias from a single-system pool.
- Under the new labels, TF-IDF is still competitive: P@5 0.975 (highest) and
  graded nDCG@10 0.821 (lowest, tied with semantic). It is not the best
  system overall.
- Old-label disagreement is 10% for TF-IDF's own results and 59–74% for
  every other system.

## 3. The improved variant: what changes

- **Name:** `top_k_average_author_discount` (`src/week4/ranking_strategy.py`).
- **Formula:** `adjusted_score = paper_score / sqrt(number_of_authors)`,
  applied before the usual top-3 average.
- **Motivation:** the error analysis found co-author tie inflation. One
  many-author paper gives all its co-authors the same score, so a Week 4 top 10
  came from only 5.5–5.9 distinct papers on average, with 6–7 of 8 top 10s
  having 3 or more researchers from one paper.
- **No tuned parameter.** The square root is a fixed choice. Note, however,
  that it was motivated by errors seen on the same 8 queries.
- **Unchanged:** the three original strategies, `WEEK4_STRATEGIES` and the
  frozen run all stay as they were (re-verified by the robustness run).

## 4. Does it improve the metrics? Not shown. It does fix the failure mode.

- **Failure mode, measured** (`artifacts/phase2/improved_variant/20261008T050358Z/coauthor_concentration.csv`):
  - The variant's top 10 come from 10.0 distinct papers per query, against
    5.5–5.9.
  - The largest group of researchers sharing one paper falls from about 3.5
    to 1.0, and no top 10 is flagged (against 6–7 of 8).
  - The q01 chatbot co-authors and the q02 code artifact leave the top 5.
  - The q07 "Editorial" remains, because it is a single-author paper.
- **Ranking quality: inconclusive.**
  - 44 of the variant's 80 results were never judged, because they were not in
    the frozen pool.
  - Condensed lists keep only 36 judged results, so its standard metrics
    (P@10 0.413, nDCG@10 0.545) are not comparable with the other systems and
    must not be read as a drop.
  - On the judged results, 33/36 = 0.917 are relevant, against 0.888–0.913
    for the six frozen systems. That is a difference of one or two
    researchers, which is not evidence of improvement.
- **To make it conclusive:** judge the 44 pairs in `unjudged_variant_pairs.csv`
  with the same rubric, then rerun `evaluate_improved_variant`.
- **Suggested wording:** "The author-count discount removes co-author
  concentration, from about 5.7 to 10 distinct evidence papers per top 10,
  without lowering precision among judged results. Its effect on standard
  metrics could not be established because 44 of its 80 results lie outside
  the judged pool."

## 5. Robustness (`artifacts/phase2/robustness/20261008T050812Z/`)

- **Setup:** 9 rephrasings of 3 benchmark queries (short keyword, detailed
  problem statement, abstract-style), plus 4 unlabelled standalone queries
  (2 ambiguous, 2 Indian-domain).
- **Main finding: all systems are sensitive to phrasing.** A rephrased query
  shares only 4–20% of its top 10 with the original (mean Jaccard).
  - Abstract-style queries change results most.
  - TF-IDF is least stable (0.04), Week 3 semantic most stable (0.20).
  - The author-discount variant (0.19) is more stable than the original
    Week 4 strategies (0.11–0.12).
- **Coverage caveat:** only 19–38% of rephrased results were judged, so the
  precision on them (0.94–1.00) is weak evidence.
- **"transformer":** every evidence title shown is about electrical power
  transformers, none about the neural architecture. The ambiguity is resolved
  to one sense silently; this is a failure case for an NLP user.
- **Indian-domain queries:** the evidence titles are on topic for all
  systems. This is a description of the titles, not a relevance label.

## 6. Assignment requirements (Week 5)

| Requirement | Status | Where |
|---|---|---|
| Quantitative evaluation | Done | P@5/10, Recall@10, MRR, nDCG@10, Hit@10 and graded nDCG@10 for 7 systems: `fair_evaluation/20261008T050903Z/`, `improved_variant/20261008T050358Z/` |
| Test set | Done, with caveats | 8 benchmark queries from Week 2 (`evaluation_candidates_labeled.csv`). No system is trained or tuned on them, but the variant was designed after inspecting errors on these same queries. The rephrased robustness queries are the only out-of-sample check. |
| Error analysis | Done | `artifacts/phase2/error_cases.csv`, `error_review_queue.csv`, `coauthor_ties.csv`, `error_analysis.md` |
| Failure cases | Done | Co-author tie inflation (25 cases); weak evidence papers (code artifact, editorial, retraction notice); the ambiguous "transformer" query; sensitivity to phrasing |
| Robustness | Done (lightweight) | `artifacts/phase2/robustness/20261008T050812Z/` |

## 7. Caveats the report must state

- The labels are AI-generated (`judge_id = Codex_AIReview`, one timestamp),
  not independent human judgments. 131 of 214 are flagged provisional,
  mostly researchers with no Week 4 evidence.
- There are only 8 queries; no significance testing was done.
- Recall@10 is computed against the judged pool, which holds about 24
  relevant researchers per query, so it cannot reach 1.0.
- The `weighted` strategy is an unimplemented placeholder and is excluded
  from every evaluation.
