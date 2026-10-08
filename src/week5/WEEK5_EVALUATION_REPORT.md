# Week 5 Evaluation Report: Evaluation and Error Analysis

**Project:** Research Discovery & Collaboration Engine (Group 7)
**Scope of Week 5:** quantitative evaluation, test set, error analysis,
robustness and failure cases for the Week 3 and Week 4 researcher-recommendation
systems.

Every number in this report comes from a file in `artifacts/`. The file is
named next to each result.

---

## 1. Project context

Given a research topic, problem statement or abstract, the system recommends
researchers whose publications match it. The data (built in Week 2) consists of
publications by authors at Indian institutions, collected through the
OpenAlex API: about 62,600 papers and 52,736 researchers.

| Week | Role |
|---|---|
| Week 2 | Data collection and preprocessing (`Data/processed/`); a small benchmark of 8 labelled queries. |
| Week 3 | First NLP pipeline (`src/week2/pipeline.py`, `src/week3/`). |
| Week 4 | Improved pipeline with paper-level retrieval (`src/week4/`). |
| Week 5 | Evaluation and error analysis of Weeks 3 and 4 (`src/week5/`, this folder). |

### 1.1 How Week 3 works

Week 3 represents **each researcher as one vector** and ranks researchers
directly. It has three modes:

- **semantic:** each paper is embedded with Sentence-BERT
  (`all-MiniLM-L6-v2`). A researcher's vector is the normalised mean of their
  paper embeddings, and the query is ranked against it by cosine similarity.
- **tfidf:** a TF-IDF vectoriser (unigrams and bigrams, English stop words)
  over each researcher's concatenated paper text, ranked by cosine similarity.
- **hybrid:** `0.7 × dense + 0.3 × sparse`, after scaling each score to [0, 1].

### 1.2 How Week 4 changes the pipeline

Week 4 searches **individual papers** instead of averaged researcher profiles:

1. Each of the 60,098 papers that has at least one author is embedded with the
   same model (`all-MiniLM-L6-v2`), and the vectors are stored in a
   persistent ChromaDB index. The 2,539 papers without authors are skipped.
2. For a query, the 100 most similar papers are retrieved.
3. The retrieved papers are grouped by author, and a **ranking strategy**
   turns each researcher's matching papers into one score.
4. The top researchers are returned with their supporting papers as
   evidence.

Because both weeks use the same embedding model, differences between them come
from the architecture (paper-level retrieval and aggregation), not from a
better model.

### 1.3 Where the Week 4 strategies live

All ranking strategies are implemented in **`src/week4/ranking_strategy.py`**
and selected through `src/week4/config.py`. None are implemented in Week 5.

| Strategy | Score of a researcher | Status |
|---|---|---|
| `best_paper` | Their single best-matching paper | Week 4; evaluated |
| `average` | Mean over all their retrieved papers | Week 4; evaluated |
| `top_k_average` | Mean of their 3 best papers | Week 4; evaluated |
| `top_k_average_author_discount` | Like `top_k_average`, but each paper score is divided by √(number of authors) | Added in Week 5 (to `src/week4`); evaluated separately (Section 6) |
| `weighted` | Formula never specified | Placeholder (`NotImplementedError`); excluded from all evaluation |

Week 5 imports Week 4's code (the recommenders, metrics and strategies). Week 5
itself contains only evaluation code, judgments, analysis, artifacts and
reports.

---

## 2. What Week 5 evaluates

Seven systems are compared. Each returns its top 10 researchers per query.

- Week 3 semantic, Week 3 TF-IDF and Week 3 hybrid.
- Week 4 `best_paper`, Week 4 `average` and Week 4 `top_k_average`.
- Week 4 `top_k_average_author_discount`, the improved variant.

**Frozen rankings.** The first six systems were run once and frozen, with their
scores and evidence, in `artifacts/runs/20261007T154203Z/`. A second
independent run (`20261007T154330Z`) was byte-for-byte identical (see
`artifacts/reproducibility/`), and both reproduce Week 4's saved results. All
later evaluation reads these frozen rankings, so **no ranking changes between
the old and new results; only the relevance labels do.**

**Metrics:**
- Precision@5 and Precision@10;
- Recall@10;
- MRR;
- nDCG@10 (binary);
- Hit@10;
- graded nDCG@10, which uses the 0–3 relevance levels as gains.

---

## 3. Test set

- **Queries:** the 8 benchmark queries from Week 2
  (`Data/processed/calculated/evaluation_candidates_labeled.csv`):
  1. natural language processing
  2. machine learning
  3. transformer based text classification
  4. information retrieval
  5. knowledge graphs
  6. battery materials and energy storage
  7. quantum computing
  8. deep learning neural networks
- **Held out:** no system is trained or tuned on these queries. All systems
  use pre-trained embeddings or TF-IDF without fitted parameters.
- **Caveat:** the improved variant (Section 6) was designed after inspecting
  errors on these same 8 queries. The robustness queries (Section 7) are the
  only out-of-sample check.
- **Judged pool:** for every query, the top 10 of all six original systems
  were pooled. This gives **214 unique query–researcher pairs** (20–33 per
  query), and every pair was judged (Section 4.2).

---

## 4. Relevance labels

### 4.1 Why the old `existing_label` benchmark is biased

The Week 2 benchmark (`existing_label`) contains 80 researchers, 10 per query,
all marked relevant (`docs/benchmark_audit.md`). Problems:

- **The 10 per query are exactly Week 3 TF-IDF's own top 10.** TF-IDF
  therefore scores 1.0 on every metric by construction.
- There are no negative labels. Every researcher outside TF-IDF's top 10 is
  unjudged, yet the evaluation counted it as **not relevant**.
- Every other system is penalised for each relevant researcher that TF-IDF
  did not return. This is pooling bias from a single-system pool.

Under these labels (`improved_variant/20261008T050358Z/old_label_summary.csv`),
TF-IDF gets 1.0 everywhere, Week 3 semantic gets P@10 0.20 and MRR 0.64, and
Week 4 gets P@10 0.19–0.20 and MRR 0.35. That made Week 4 look worse than
Week 3.

### 4.2 How `manual_relevance` was created and used

- **Template:** `judging/templates/judging_template.csv` lists all 214 pooled
  pairs with each researcher's evidence papers and profile paper titles.
- **Rubric** (`judging/README.md`):
  - 3 = highly relevant;
  - 2 = relevant;
  - 1 = weakly or indirectly relevant;
  - 0 = not relevant;
  - blank = unjudged, never read as 0.

  Levels 2 and 3 count as relevant for the binary metrics.
- **Labels:** `judging/manual_judgments_completed.csv`, with all 214 pairs
  judged: 115 × level 3, 77 × level 2, 16 × level 1 and 6 × level 0. It
  passes the validator, which checks that no frozen column was altered.
  `judging/manual_judgments.csv` is the untouched blank template copy.
- **131 rows flagged provisional.**
  - 127 of them have no Week 4 evidence: the researcher was returned only by
    Week 3, whose evaluation adapter exposes no paper evidence. These were
    judged from profile paper titles.
  - 4 have weak evidence: a code artifact, two retraction notices and an
    editorial.
- **How the labels are used:** metrics use `manual_relevance` only. The old
  `existing_label` is used only for the label comparison.

> **Important caveat:** the labels are **AI-assisted, not independently
> human-labelled**. They were produced by an AI reviewer
> (`judge_id = Codex_AIReview`, a single `judged_at` timestamp) following the
> written rubric. No second annotator exists, so inter-annotator agreement
> cannot be reported. A human check of at least a sample is advisable before
> treating these labels as ground truth.

### 4.3 Old vs new labels (`artifacts/phase2/label_agreement.md`)

| Slice | Pairs | Labels differ | Disagreement |
|---|---|---|---|
| Old positives (`existing_label = 1`) | 80 | 8 | 10.0% |
| Old implied negatives (blank, previously scored as wrong) | 134 | 120 | 89.5% |
| All pooled pairs | 214 | 128 | 59.8% |

The old positives mostly hold up: 72 of 80 are judged relevant. However, 120
of the 134 researchers that the old benchmark silently counted as wrong are
judged relevant. By system, disagreement is 10% for TF-IDF's own results and
59–74% for every other system.

---

## 5. Quantitative results: fair evaluation of the six original systems

Source: `artifacts/phase2/fair_evaluation/20261008T050903Z/summary.csv`. It is
identical to the first Phase 2 run, `20261007T172845Z`. All 8 queries are fully
judged, so no ranking contains an unjudged researcher.

| System | P@5 | P@10 | Recall@10 | MRR | nDCG@10 | Hit@10 | nDCG@10 (graded) |
|---|---|---|---|---|---|---|---|
| Week 3 semantic | 0.900 | 0.900 | 0.387 | 0.875 | 0.887 | 1.0 | 0.821 |
| Week 3 TF-IDF | **0.975** | 0.900 | 0.382 | **1.000** | **0.928** | 1.0 | 0.821 |
| Week 3 hybrid | 0.900 | **0.913** | 0.401 | **1.000** | 0.921 | 1.0 | 0.858 |
| Week 4 best_paper | 0.900 | 0.888 | 0.397 | **1.000** | 0.898 | 1.0 | 0.874 |
| Week 4 average | 0.875 | 0.900 | **0.401** | **1.000** | 0.901 | 1.0 | **0.882** |
| Week 4 top_k_average | 0.875 | 0.900 | **0.401** | **1.000** | 0.901 | 1.0 | **0.882** |

**Reading the table:**
- **The six systems are close.** Week 4 is **on par** with the Week 3
  baselines, not worse as the old labels suggested and not clearly better.
- **Week 4's gains are modest:**
  - MRR 1.0 (a relevant researcher at rank 1 for every query), versus 0.875
    for Week 3 semantic;
  - the highest graded nDCG@10 (0.874–0.882 versus 0.821–0.858), meaning
    more *highly* relevant (level 3) researchers near the top.
- **Week 3 TF-IDF** is still strong on P@5 and binary nDCG@10. Its earlier
  perfect score, however, was an artefact of the labels.
- **Recall@10** is low for every system because the judged pool holds about
  24 relevant researchers per query, and only 10 can be returned.
- **`average` and `top_k_average`** produce identical rankings on all 8
  queries.
- **No significance testing** was done: there are only 8 queries, and the
  labels are AI-assisted.

**Why Week 4 is not "suddenly improving":** the Week 4 rankings are the same
frozen rankings that scored P@10 0.19 under the old labels. Only the
measurement changed. Under labels that do not favour TF-IDF, Week 4 is
measured fairly for the first time.

A sensitivity run that treats flagged rows as unjudged
(`fair_evaluation/20261007T172845Z-exclude-flagged/`) is **not** a fair
comparison. Most flagged rows are Week 3-only researchers, so excluding them
favours Week 4 by construction.

---

## 6. Improved variant: `top_k_average_author_discount`

### 6.1 Motivation and change

The error analysis (Section 8) found **co-author tie inflation**. A paper with
many authors gives every co-author the same score, so the top 10 of a Week 4
system is filled from very few papers:

- 19 of 24 Week 4 system–query rankings are flagged;
- one top 10 comes from only 3 distinct papers.

The variant changes one line of the scoring:

```
adjusted_score = paper_score / sqrt(number_of_authors_of_that_paper)
researcher_score = mean of the researcher's 3 best adjusted scores
```

- Single-author papers are unchanged.
- Co-authors of one paper still share equal credit, but a 9-author paper now
  counts one third as much per author.
- There is no tuned parameter.
- The code is in `src/week4/ranking_strategy.py`
  (`TopKAverageAuthorDiscountStrategy`) and is unit-tested in
  `tests/test_variant.py`.
- The three original strategies are untouched, and the frozen run still
  reproduces.

### 6.2 Results (`artifacts/phase2/improved_variant/20261008T050358Z/`)

The variant is scored on the same 8 queries and the same judged pool. The
variant returns 44 researchers (out of 80 results) that no original system
returned, so none of the 214 judgments covers them. These are **removed from
its lists, not counted as wrong** (condensed lists).

| System | P@5 | P@10 | nDCG@10 | nDCG@10 (graded) | Unjudged removed | Judged results in top 10 | Precision among judged |
|---|---|---|---|---|---|---|---|
| Week 4 top_k_average | 0.875 | 0.900 | 0.901 | 0.882 | 0 | 80 | 0.900 |
| Week 4 author_discount | 0.725 | 0.413 | 0.545 | 0.517 | 44 | 36 | 0.917 |
| Range of the six original systems | | | | | 0 | 80 each | 0.888–0.913 |

Co-author concentration (`coauthor_concentration.csv`, per top 10, mean over 8
queries):

| System | Distinct evidence papers | Largest group sharing one paper | Top 10s with ≥ 3 researchers on one paper |
|---|---|---|---|
| Week 4 best_paper | 5.5 | 3.75 | 7 of 8 |
| Week 4 average / top_k_average | 5.9 | 3.5 | 6 of 8 |
| **Week 4 author_discount** | **10.0** | **1.0** | **0 of 8** |

### 6.3 Interpretation

- **It fixes co-author concentration.** Every top 10 now comes from 10
  different papers. Two documented failures (F1 and F2 below) leave the
  top 5.
- **It does not yet prove better accuracy.**
  - The variant's condensed lists keep only 2–7 judged researchers per query,
    which mechanically lowers P@10 and nDCG@10. **Those numbers are not a
    drop in quality and are not comparable** with the other systems.
  - On the 36 judged results, 33 are relevant (0.917), against 0.888–0.913
    for the original systems. That is a difference of one or two researchers.
- **To settle it:** judge the 44 pairs listed in
  `improved_variant/20261008T050358Z/unjudged_variant_pairs.csv`, then rerun
  the variant evaluation. They were deliberately left blank rather than
  guessed.

---

## 7. Robustness (`artifacts/phase2/robustness/20261008T050812Z/`)

All seven systems were run on 13 query texts written for this check. Before
the check, the base queries were re-run; they reproduced the frozen rankings
exactly (`run.json`).

- **Rephrasings (9 queries):** q01, q06 and q07 were each rewritten as a
  short keyword ("NLP", "batteries", "quantum"), a detailed problem statement
  and an abstract-style paragraph.
  - Each result list is compared with the same system's list for the
    original query (top-10 Jaccard overlap).
  - Each list is also checked against the original query's judgments, because
    the information need is the same. Researchers not judged for the original
    query count as unjudged, not wrong.
- **Standalone queries (4, unlabelled):**
  - ambiguous: "transformer" and "network";
  - Indian-domain: "crop yield prediction for Indian agriculture using monsoon
    rainfall data" and "speech recognition for Indian languages such as Hindi
    and Marathi".
  - Only the evidence titles are inspected; no relevance is claimed.

| System | Mean top-10 Jaccard with original | Same top-1 | Rephrased results that are judged | Precision among judged |
|---|---|---|---|---|
| Week 3 semantic | **0.200** | 1 of 9 | 38% | 0.94 |
| Week 3 TF-IDF | 0.042 | 0 of 9 | 19% | 1.00 |
| Week 3 hybrid | 0.062 | 1 of 9 | 31% | 1.00 |
| Week 4 best_paper | 0.109 | 0 of 9 | 23% | 0.95 |
| Week 4 average | 0.118 | 0 of 9 | 21% | 0.95 |
| Week 4 top_k_average | 0.113 | 0 of 9 | 21% | 0.95 |
| Week 4 author_discount | 0.193 | 0 of 9 | 21% | 0.95 |

**Findings:**
- **Every system is sensitive to phrasing.** A rephrased query shares only
  4–20% of its top 10 with the original.
- By query type, abstract-style rephrasings change results most (mean
  Jaccard 0.06), and short keywords least (0.16).
- **TF-IDF is the least stable** (0.04), as expected for exact term
  matching. Week 3 semantic and the author-discount variant are the most
  stable.
- **The judged researchers that recur are mostly relevant** (0.94–1.00).
  However, only 19–38% of rephrased results are judged, so this does not show
  that rephrased rankings are as good as the originals.
- **The Indian-domain queries** return topically matching evidence titles
  wherever titles are available: monsoon rainfall and crop yield in Indian
  states, and Hindi, Marathi and Odia speech. Week 3 TF-IDF's top 5 for the
  speech query has no evidence titles.
- **The ambiguous queries** are failure cases (F8 below).

---

## 8. Error and failure cases

Sources:
- `artifacts/phase2/error_cases.csv` (25 confirmed co-author tie cases,
  E001–E025);
- `error_review_queue.csv` (top-5 results with their judgments);
- `coauthor_ties.csv`;
- the robustness artifacts.

Levels are `manual_relevance`.

| # | Query | System(s) | What happened | Likely cause | Effect of the variant |
|---|---|---|---|---|---|
| F1 | q01 natural language processing | Week 4 all three (E001, E010, E018); Week 3 semantic | Ranks 3–7 (best_paper) are five co-authors of one chatbot paper, "Virtual Assistant for Appointment Booking" (levels 2, 1, 1, 1, 1). Week 3 semantic also ranks two of them #4 and #5. | Co-author tie inflation: one weakly related paper gives five researchers the same score. | Fixed: none in its top 5. |
| F2 | q02 machine learning | Week 3 semantic #1; Week 4 all #2 | Parthiv Sarkar (level 1), whose evidence is "Code for the ML models", a code artifact, not a research paper. | No document-type filtering; the title shares the query words. | Fixed: leaves the top 5. |
| F3 | q02 machine learning | Week 4 best_paper (E003) | Ranks 7–9 are co-authors of "Retraction Note to: A Survey on Applications of Machine Learning Algorithms in Health care" (levels 1, 2, 1). | A retraction notice repeats the original title, so it matches well; no filtering of notices. | Fixed: none of the three in its top 10. |
| F4 | q07 quantum computing | Week 3 semantic #1; Week 4 all #2 | Jeya Mala (level 1); evidence paper titled "Editorial". | A non-research document is indexed like a paper; there is no document-type filtering. | **Not fixed** (still #2): single-author paper, so the discount does nothing. |
| F5 | q05 knowledge graphs | Week 4 all three (E006, E014, E022 and E007, E015, E023) | Ranks 1–5 are co-authors of one survey ("Knowledge Graphs and AI: A Survey…"), and ranks 7–10 co-authors of another. The top 10 comes from only 3 papers. All are judged 3. | Co-author tie inflation. Not a precision error, but a diversity failure: a user sees roughly three research teams. | Fixed: 10 distinct papers. |
| F6 | q08 deep learning neural networks | Week 4 all three (E009, E017, E025) | Ranks 6–10 are five co-authors of one survey on loss functions in computer vision (all level 3). | Co-author tie inflation; one survey with many authors dominates. | Fixed. |
| F7 | q02 machine learning | Week 4 author_discount | The variant brings in Satish Gajawada at rank 3 (level 1; evidence "Excellent Artificial Intelligence (EAI)"). | The discount promotes single-author papers, including weakly related ones. | Introduced by the variant. |
| F8 | "transformer" (robustness) | All systems | Every evidence title shown refers to **electrical power transformers**, none to the neural architecture. Original Week 4 strategies put four co-authors of "Energization of Power Transformer from Low Power Rating Grid-Forming VSI under Black Start Scenario" at ranks 1–4. | Ambiguous query; the system silently picks one sense, with no disambiguation or diversification. | Same sense, but 5 distinct papers in its top 5. |
| F9 | Rephrasings of q01 and q06 (robustness) | Week 3 hybrid and TF-IDF; Week 4 original strategies | The abstract-style version of "natural language processing" shares **0 of 10** researchers with the original for Week 3 hybrid and all three original Week 4 strategies. The abstract-style battery query shares 0 of 10 for TF-IDF and all three original Week 4 strategies. | Likely: generic abstract wording ("we propose", "outperforms baselines") dilutes the topic terms in both the embedding and TF-IDF. | Partly: 0.11 overlap for q01, 0.00 for q06. |
| F10 | All queries | Week 3 (all modes) | 127 of 214 pooled pairs have no paper evidence, because Week 3 returned the researcher but its evaluation adapter exposes no supporting papers. These had to be judged from profile titles and are flagged provisional. | Week 3 returns researchers without evidence, so its results are hard to verify or explain. | Not applicable (Week 4 always returns evidence). |

Summary of failure types:
- co-author concentration (F1, F5, F6);
- weak or non-research evidence documents (F2, F3, F4);
- side effects of the fix (F7);
- query ambiguity (F8);
- sensitivity to phrasing (F9);
- missing explainability in Week 3 (F10).

---

## 9. Limitations

- **AI-assisted labels.** One automated reviewer, no human verification and
  no agreement measure. 131 of 214 labels are flagged provisional.
- **Small test set.** 8 queries, with no significance testing. Differences of
  a few hundredths between systems should not be over-interpreted.
- **Pool-bound recall.** The judged pool contains only the six original
  systems' top 10s. Recall@10 is relative to that pool, and the variant has
  44 unjudged results.
- **Small robustness check.** 13 hand-written queries. Rephrasings reuse
  the original query's judgments and have low judged coverage; standalone
  queries have no labels.
- **Variant designed on the test queries.** The variant was designed after
  seeing errors on the same 8 test queries.
- **Excluded strategy.** `weighted` remains an unimplemented placeholder and
  is excluded.

## 10. Next steps

1. Judge the 44 pairs in `unjudged_variant_pairs.csv`, then rerun the variant
   evaluation to measure its accuracy fairly.
2. Have a human check a sample of the AI-assisted labels (especially the 131
   flagged rows) and report agreement.
3. Filter non-research documents (editorials, retraction notices, code
   deposits) before ranking (F2–F4).
4. Diversify results for ambiguous queries, or ask the user to clarify (F8).
5. Extend the benchmark with more queries and natural rephrasings, including
   abstract-style queries (F9).
6. Expose paper evidence from the Week 3 baseline so all systems are equally
   explainable (F10).

## 11. Conclusion

- **The old benchmark was the main problem.** Its labels were Week 3
  TF-IDF's own top 10, which made TF-IDF perfect and made Week 4 look worse
  than Week 3.
- **Under fair labels the systems are close.** All six original systems
  score P@10 0.89–0.91 on the same frozen rankings, judged over the full
  214-pair pool.
- **Week 4 matches Week 3 and leads slightly where it counts most:** a
  relevant researcher at rank 1 for every query (MRR 1.0) and the highest
  graded nDCG@10 (0.874–0.882). On this small test set, Week 4 is a
  competitive, more explainable alternative (it always returns supporting
  papers). It is not a clear accuracy improvement.
- **The improved variant fixes the main Week 4 failure, co-author
  concentration:** 10 distinct papers per top 10 instead of about 5.7. Its
  accuracy benefit is not yet established, because 44 of its results are
  unjudged.
- **Robustness is the weakest area for every system.** Rephrasing a query
  changes most of the top 10, and ambiguous queries are resolved silently to
  one sense.

---

## Folder guide (`src/week5/`)

| Folder | Contents |
|---|---|
| `judging/` | Relevance judging: rubric (`README.md`), frozen template, the blank `manual_judgments.csv`, the completed `manual_judgments_completed.csv`, and the validator/merge code. |
| `evaluation/` | Frozen-run creation, reproducibility check, evidence records, fair evaluation (judged pairs only), and rankings for the improved variant. |
| `analysis/` | Pool summary, label agreement (old vs new), error analysis (co-author ties, error cases, review queue) and the robustness check. |
| `scripts/` | Command-line entry points for everything above (Section "How to rerun"). |
| `artifacts/` | Generated outputs. `runs/` holds the frozen rankings; `reproducibility/` the run comparison; `phase2/` the fair evaluation, label agreement, error analysis, improved variant and robustness results, each in a dated folder. |
| `docs/` | `benchmark_audit.md` (old-label bias), `phase2_status.md` (detailed results), `report_notes.md` (summary notes). |
| `tests/` | Unit tests for judging, fair evaluation, the variant and robustness measures. |

## How to rerun

From the repository root (`SNLP_Project/`), with dependencies installed
(`uv sync`):

```bash
uv run python -m src.week5.scripts.validate_judgments         # check the 214 judgments
uv run python -m src.week5.scripts.fair_evaluation            # Section 5 -> artifacts/phase2/fair_evaluation/<time>/
uv run python -m src.week5.scripts.compare_labels             # Section 4.3 -> artifacts/phase2/label_agreement.*
uv run python -m src.week5.scripts.evaluate_improved_variant  # Section 6 -> artifacts/phase2/improved_variant/<time>/
uv run python -m src.week5.scripts.robustness_check           # Section 7 -> artifacts/phase2/robustness/<time>/
uv run python -m unittest discover -s src/week5/tests -t .    # unit tests
```

The first three commands and the tests read the frozen artifacts, the
judgments and the Week 2 data. The variant and robustness scripts rank
queries live, so they also need the Week 4 ChromaDB index
(`python -m src.week4.build_chromadb`, about 1.2 GB in `Data/week4/chroma/`,
not included in the repository).

## Dependencies and what to submit

- **Week 4 methods are implemented in `src/week4`**, including the improved
  variant (`src/week4/ranking_strategy.py`). Week 5 only evaluates them.
- **Week 5 depends on Week 4 code.** Its scripts import the recommenders,
  metrics and strategies from `src/week4`. Through Week 4's baseline adapter,
  they also depend on the Week 3 pipeline (`src/week2/pipeline.py`) and the
  Week 2 data.
- **Zipping only `src/week5`** gives a complete record of the evaluation:
  this report, the judgments, the frozen rankings, all metrics and the error
  cases. It is readable on its own, but **not runnable**. Running it requires
  the full repository (`src/week2`–`src/week5`, `Data/` and `pyproject.toml`),
  plus the ChromaDB index for the live-ranking scripts.
