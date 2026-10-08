# Pre-submission audit (Week 5)

Read-only audit made on 2026-10-08 before zipping. No code, pipeline file,
metric, label, artifact or report was changed while preparing it.

> This is an internal checklist. It lives inside `src/week5/`, so it will end
> up in the zip unless you delete it or exclude it (see Section 6).

## 1. Executive summary

- **Requirements:** all Week 5 requirements are covered by
  `WEEK5_EVALUATION_REPORT.md` (Section 2).
- **Must fix before submission (two items):**
  1. The report's conclusion claims Week 4 "leads" on MRR, but Week 3 TF-IDF
     and hybrid also score MRR 1.0 (Section 4, H1).
  2. The report calls the labels "AI-assisted", while every other document
     says "AI-generated". "AI-generated" is the accurate term, because no
     human judged any row (H2).
- **Provenance:** otherwise honest. The AI origin of the labels, the
  `Codex_AIReview` judge ID and the variant's unproven accuracy are all
  stated clearly.
- **Frontend:** Streamlit (`app.py`). It loads and runs a search with no
  errors, and Week 5 did not touch it. Two non-blocking inconsistencies
  (Section 5).
- **Writing:** correct but visibly machine-formatted in places: heavy bold,
  templated "Reading the table" blocks, phrasing that echoes the task prompt,
  and three documents repeating the same numbers. It needs a human pass
  (Section 3).
- **Zip:** submit `src/week4` + `src/week5`, without caches, `.DS_Store`
  files or data and index files (Section 6).

## 2. Requirement coverage

| Requirement | Status | Where |
|---|---|---|
| Quantitative evaluation | Met | Report §5 (6 systems × 7 metrics) and §6 (variant); `artifacts/phase2/fair_evaluation/20261008T050903Z/` |
| Test set | Met | Report §3: 8 Week 2 queries, 214 judged pairs, and the caveat that the variant was designed on them |
| Error analysis | Met | Report §8; `artifacts/phase2/error_cases.csv` (25 cases), `error_review_queue.csv`, `coauthor_ties.csv` |
| Robustness | Met (lightweight) | Report §7; `artifacts/phase2/robustness/20261008T050812Z/` |
| Failure cases | Met | Report §8, F1–F10 |
| At least 5–10 concrete error cases | Met | 10 cases (F1–F10), each with query, system, rank, level, evidence title and cause; 25 more in `error_cases.csv` |
| Metrics table | Met | Report §5 and §6.2; old-vs-new label table in §4.3 |
| Limitations and conclusion | Met | Report §9 and §11 (§11 needs the H1 fix) |

## 3. AI-authorship / wording concerns

Nothing here is factually wrong (factual issues are in Section 4). These are
places where the text reads as generated or templated. The aim of the
suggestions is plainer, more personal writing in your group's own voice, not
hiding how the work was produced (see H6).

| File | Section | Concern | Suggested revision direction |
|---|---|---|---|
| `WEEK5_EVALUATION_REPORT.md` | Whole file | About 85 bold phrases in 4,059 words; nearly every bullet opens with a bold label. Reads like a formatted summary, not a report. | Keep bold for at most 5–6 key findings. Turn the bullet lists in §5, §6.3 and §11 into short paragraphs. |
| `WEEK5_EVALUATION_REPORT.md` | Title | "Week 5 Evaluation Report: Evaluation and Error Analysis" repeats itself. | E.g. "Week 5: Evaluating the Researcher Recommender". |
| `WEEK5_EVALUATION_REPORT.md` | §5, "Why Week 4 is not 'suddenly improving'" | The quoted phrase comes from our own instructions, so it answers a question the reader never asked; it reads defensive. | State it once, plainly: "The rankings did not change; only the labels did." Drop the heading-style lead-in. |
| `WEEK5_EVALUATION_REPORT.md` | §5, "Reading the table" | Templated label plus six parallel bold bullets. | Two or three sentences on what you actually noticed in the table. |
| `WEEK5_EVALUATION_REPORT.md` | §4.2, "Important caveat" box | The bold blockquote is correct but formal and boilerplate-like. | Say it in the group's voice: who decided to use an AI reviewer, why, and what you would do with more time. |
| `WEEK5_EVALUATION_REPORT.md` | §8 table | Very dense 6-column table with long cells; uniform, mechanical structure. | Keep the table for F1–F10 IDs and short labels; describe the 3–4 most interesting cases (F1, F4, F8, F9) in prose. |
| `WEEK5_EVALUATION_REPORT.md` | Folder guide, How to rerun | Fine for a README; reads like documentation inside a report. | Optionally move both to `README.md` and link to it. |
| `docs/report_notes.md` | Whole file | Notes addressed to "the report", with "Suggested wording" in quotes. If submitted, it looks like a drafting aid. | Remove it from the zip, or retitle it "Working notes". |
| `docs/phase2_status.md`, `docs/report_notes.md`, report | Overlap | The same numbers and tables appear three times. Wording differences between copies already caused H2. | Make the report the single source; mark the other two as working notes, or exclude `report_notes.md`. |
| `README.md` | "Out of scope for Phase 1" | "Report writing (Phase 3)" is stale: there is no Phase 3, and the report now exists. | Update or delete that bullet. |

## 4. Honesty and provenance checks

| Check | Result | Evidence |
|---|---|---|
| `manual_relevance` labels described as AI-produced | Yes | Report §4.2, §9; `phase2_status.md` "Labels"; `README.md` |
| `judge_id = Codex_AIReview` stated | Yes | Report §4.2; `phase2_status.md`; `report_notes.md` §7; `README.md` |
| States the labels are not independently human-annotated | Yes | Report §4.2 ("not independently human-labelled"), §9 ("no human verification") |
| Flagged rows need optional human review | Yes | Report §4.2 (131 provisional rows), §10 step 2 |
| Variant not proven more accurate (unjudged pairs) | Yes | Report §6.3 and §11 (44 of 80 unjudged); `unjudged_variant_pairs.csv` is left blank |
| Week 4 methods live in `src/week4` | Yes | Report §1.3 and "Dependencies and what to submit" |
| Week 5 depends on Week 4 code/outputs | Yes | Report §1.3 and "Dependencies…"; all scripts import `src.week4` |

Issues found:

- **H1 (must fix): MRR overclaim.**
  - **Where:** report §11, "Week 4 matches Week 3 and leads slightly where it
    counts most: … (MRR 1.0)". §5's "Week 4's gains are modest" compares MRR
    only against Week 3 semantic. The same selective comparison appears in
    `report_notes.md` §1.
  - **Problem:** Week 3 TF-IDF and hybrid also have MRR 1.0, so Week 4
    leads only on graded nDCG@10.
  - **Fix:** say Week 4 ties TF-IDF and hybrid on MRR and is highest only on
    graded nDCG@10. Also drop "where it counts most", which is an
    unsupported value judgment.
- **H2 (must fix): "AI-assisted" vs "AI-generated".**
  - **Where:** the report (§4.2, §5, §9, §10) says "AI-assisted";
    `phase2_status.md`, `report_notes.md` and `README.md` say "AI-generated".
  - **Problem:** every row was judged by the AI reviewer and none by a
    person, so "AI-assisted" understates it.
  - **Fix:** use "AI-generated (no human review)" consistently.
- **H3 (minor): "TF-IDF without fitted parameters".**
  - **Where:** report §3.
  - **Problem:** TF-IDF is fitted on the researcher profiles (vocabulary and
    IDF weights), and the hybrid weight α = 0.7 is a fixed Week 3 choice.
  - **Fix:** say "nothing is fitted or tuned on the benchmark queries".
- **H4 (minor): "The old benchmark was the main problem".**
  - **Where:** report §11.
  - **Problem:** this is fair for the measurement, but phrased as a verdict.
  - **Fix:** "Most of the earlier gap between systems came from the
    benchmark labels, not from the systems."
- **H5 (OK, keep):** the robustness and standalone-query findings are worded
  as observations of evidence titles, with no relevance claimed. The
  exclude-flagged run is correctly called "not a fair comparison".
- **H6 (your decision): disclosure of AI tool use.**
  - **Situation:** the Week 5 code, documents and this report were drafted
    with an AI coding assistant, in addition to the AI-generated labels. The
    documents currently disclose only the labels.
  - **Action:** check your course's AI-use policy. If it asks for
    disclosure, add one sentence to the report saying which parts were
    AI-assisted and what the group did itself (design decisions, review,
    verification).

## 5. Frontend status

> **Update (2026-10-08, after this audit):** the frontend has since been
> migrated from Streamlit to Flask with Jinja2 templates (`app.py`,
> `services.py`, `templates/`, `static/`); run it with `uv run python app.py`.
> The table below describes the Streamlit version as audited. F-2 no longer
> applies; F-1 still does, because the benchmark page shows the same CSV files.

| Check | Result |
|---|---|
| Framework | **Streamlit** (`import streamlit as st`; `streamlit>=1.40.0` in `pyproject.toml`; 1.64.0 installed) |
| Entry point | `app.py` at the repository root; run with `uv run streamlit run app.py` (root `README.md` §3) |
| Static check | `app.py` compiles. A headless `streamlit.testing.AppTest` run loaded the pipeline, rendered all 3 tabs, and ran a search ("quantum computing": 10 researchers) and "Compare Models", with no exceptions or errors. |
| Backend used | Only the Week 3 pipeline (`src/week2/pipeline.py`: semantic, TF-IDF, hybrid). Week 4 and Week 5 are not used by the UI. |
| Affected by Week 5? | **No.** `app.py`, `pyproject.toml` and `src/week2` have no uncommitted changes; `app.py` last changed in commit `4cf053a`. Week 5's only code change outside `src/week5` is the new strategy in `src/week4`, which the app does not import. |

Inconsistencies (non-blocking):

- **F-1: stale results on the "Benchmark & Metrics" tab.** It shows
  `Data/processed/calculated/model_comparison.csv`, where every Week 3 mode
  scores 1.0. Week 4's README and the Week 5 report explain why those numbers
  are an artefact. A grader who opens the app sees results that contradict
  the report.
  - **Option (code change, only if you want it):** a one-line caption on that
    tab pointing to the Week 5 report.
- **F-2: deprecation warnings.** `st.dataframe(..., use_container_width=True)`
  and the search button log warnings that `use_container_width` will be
  removed. It still works with the installed 1.64.0, but a future Streamlit
  upgrade could break it.
- **F-3: root README covers only Week 3.** `README.md` describes the
  Week 3 deliverable and does not mention Weeks 4–5. This only matters if the
  root README is submitted.

## 6. Zip recommendation

**Submit `src/week4` + `src/week5`**, not `src/week5` alone:

- the improved variant is implemented in `src/week4/ranking_strategy.py`
  and `src/week4/config.py`, so a grader can't see it in `src/week5`;
- every Week 5 script and test imports `src.week4`;
- the report cites `src/week4` paths throughout.

`src/week5` alone is fully readable (report, judgments, frozen rankings,
metrics), but not runnable, and the variant's code would be missing. Neither
zip is runnable on its own; that needs the whole repository plus `Data/` and
the index. The report states this.

Exclude:

| Item | Why |
|---|---|
| `__pycache__/` (8 folders under `src/`) | Generated bytecode |
| `.DS_Store` (`src/.DS_Store`, `Data/.DS_Store`) | macOS metadata (none inside `src/week4` or `src/week5` right now) |
| `Data/week4/chroma/` (1.2 GB) | Rebuildable index; `Data/` is 3.2 GB in total |
| `Data/` in general | Not needed to read the evaluation; all evidence is in `src/week5/artifacts/` |
| `src/week5/docs/pre_submission_audit.md` (this file) | Internal checklist |
| `src/week5/docs/report_notes.md` (optional) | Drafting notes (see Section 3) |

```bash
zip -r week5_submission.zip src/week4 src/week5 \
  -x "*/__pycache__/*" "*.DS_Store" "src/week5/docs/pre_submission_audit.md"
```

Add `"src/week5/docs/report_notes.md"` to the `-x` list if you decide not to
submit it. If the frontend is part of the submission, add `app.py`.

## 7. Final action checklist

- [ ] Fix H1: MRR wording in report §5 and §11 (and in `report_notes.md` §1
      if you submit it).
- [ ] Fix H2: use "AI-generated" consistently for the labels.
- [ ] Optionally fix H3 and H4 (one sentence each).
- [ ] Decide on H6: add an AI-use disclosure if your course policy asks for
      one.
- [ ] Do a human editing pass on the report using Section 3. Priorities:
      less bold, prose instead of bullets in §5, §6.3 and §11, and your own
      voice for the label caveat.
- [ ] Spot-check a sample of `manual_judgments_completed.csv`, especially
      flagged rows and the 22 rows at levels 0–1, so you can defend the
      labels if asked.
- [ ] Update the stale "Report writing (Phase 3)" bullet in `README.md`.
- [ ] Decide whether to submit `docs/report_notes.md`.
- [ ] Decide whether to address F-1 (stale app metrics); this is optional
      and needs a code change.
- [ ] Zip with the command in Section 6, then open the zip and check that it
      contains `src/week4/ranking_strategy.py` and
      `src/week5/WEEK5_EVALUATION_REPORT.md`, and no `__pycache__`.
- [ ] Do not commit or push until you have reviewed everything (you push
      manually).
