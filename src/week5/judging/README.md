# Manual Relevance Judging (Week 5, Phase 2)

The old benchmark only labels TF-IDF's own top 10 per query. To compare all
six systems fairly, every researcher that any system ranked in its top 10 (the
pool: 214 query–researcher pairs over 8 queries) needs a manual relevance
judgment.

## Files

| File | Role |
|---|---|
| `manual_judgments.csv` | **The file you edit.** One row per pooled pair. |
| `templates/judging_template.csv` | Fixed reference generated from the frozen Phase 1 run. Do not edit; the validator compares against it. |

Both are UTF-8 CSV files and open in Excel, Google Sheets or LibreOffice. In
Excel, save with **File → Save As → CSV UTF-8**, so researcher names with
accents survive.

Create both (only needed once; an existing `manual_judgments.csv` is never
overwritten):

```bash
uv run python -m src.week5.scripts.prepare_judging
```

## Columns

Do not change these. They come from the frozen run:

| Column | Meaning |
|---|---|
| `query_id`, `query_text` | The benchmark query. |
| `researcher_id`, `researcher_name` | OpenAlex author ID and name. |
| `system_source` | Every system that returned the researcher, with its rank (e.g. `Week 4 (best_paper) #3`). |
| `rank` | Best rank across systems. |
| `systems` | How many of the six systems returned the researcher. |
| `score` | Each system's score (scales differ between systems, so don't compare them across systems). |
| `evidence_paper_ids`, `evidence_paper_titles` | The researcher's papers that Week 4 retrieved for this query (blank if only Week 3 returned them). |
| `profile_paper_count`, `profile_paper_titles` | All of the researcher's papers in the dataset (newest 5 shown), so every row has evidence. |
| `existing_label` | `1` if the researcher is in the old TF-IDF-derived benchmark, blank if not. Blank does **not** mean irrelevant. |

Fill these in:

| Column | What to enter |
|---|---|
| `manual_relevance` | `0`, `1`, `2` or `3` (see below), or leave blank if not judged yet. |
| `review_flag` | `yes` (or `needs_review`) if the evidence is thin, otherwise `no` or blank. A flagged row may stay unjudged or carry a provisional level; either way, explain it in `judgment_notes`. |
| `judge_id` | Your initials or name, the same on every row you judge. |
| `judgment_notes` | One short sentence citing the evidence that decided the level. Required on every judged or flagged row. |
| `judged_at` | Date of the judgment, e.g. `2026-10-08`. |

## Relevance scale

| Level | Meaning | Use when |
|---|---|---|
| 3 | Highly relevant | The evidence papers directly address the query topic. |
| 2 | Relevant | The connection to the query is clear but not central to the papers. |
| 1 | Weakly or indirectly relevant | The connection is indirect or only partly related. |
| 0 | Not relevant | The evidence does not show a meaningful connection to the query. |
| blank | Unjudged | Not judged yet. Blank is never read as 0. |

For binary metrics, levels 2 and 3 count as relevant.

Rules:

- Judge from the papers (`evidence_paper_titles` first, then
  `profile_paper_titles`) against the query, not from the researcher's name.
- A generic shared word is not enough for a high level. For example, a paper
  that only mentions "learning" is not, by itself, evidence for the query
  "machine learning".
- Judge each pair on its own merits, regardless of `system_source`, `score`
  or `existing_label`.
- If the titles are not enough to decide, set `review_flag` to `yes` and say
  what is missing in `judgment_notes`. Either leave `manual_relevance` blank
  or give a clearly provisional level; never guess silently.

Illustrative example (fictional, not part of the data):

| query_text | evidence_paper_titles | manual_relevance | judgment_notes |
|---|---|---|---|
| quantum computing | 2024: Error correction for superconducting qubits | 3 | Paper is directly about quantum computing hardware. |
| quantum computing | 2023: Quantum dots for solar cells | 1 | "Quantum" refers to materials physics, not computation. |

## Validate, merge, evaluate

The completed annotation is saved as `manual_judgments_completed.csv`. When
that file exists, every script below uses it (and prints its path);
`manual_judgments.csv` stays the untouched blank copy. Metrics always use
`manual_relevance`; `existing_label` is only used by `compare_labels`.

```bash
uv run python -m src.week5.scripts.validate_judgments   # checks the file; exit status 1 on any error
uv run python -m src.week5.scripts.merge_judgments      # attaches judgments to the frozen rankings
uv run python -m src.week5.scripts.fair_evaluation      # metrics on judged queries only
uv run python -m src.week5.scripts.compare_labels       # old existing_label vs new manual_relevance
uv run python -m src.week5.scripts.summarize_pool       # judging progress
uv run python -m src.week5.scripts.analyze_errors       # error tables, updated with your labels
```

The validator rejects:
- missing columns;
- levels other than blank/0/1/2/3;
- `review_flag` values other than blank, `no`, `yes` or `needs_review`;
- judged rows without `judge_id`, `judgment_notes` or an ISO `judged_at`;
- flagged rows with no note;
- duplicate pairs, or pairs added or removed;
- any change to a frozen column.

Merge, fair evaluation, pool summary and error analysis all refuse to run on
a file that does not validate.

## Fair evaluation rules

- By default, a query is evaluated only once all of its pooled pairs are
  judged (`--min-judged-fraction 1.0`). Then no ranking contains an unjudged
  researcher, because the pool holds every system's top 10.
- With a lower threshold, unjudged researchers are removed from the rankings
  before scoring, never counted as wrong. The output reports how many were
  removed.
- Queries where no pooled researcher is judged 2 or 3 are excluded and
  reported.
- Rankings are the frozen Phase 1 rankings; only the labels change.
