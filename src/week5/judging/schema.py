"""Columns, relevance scale and file format of the manual judging file."""

from pathlib import Path

import pandas as pd

RELEVANCE_SCALE: dict[int, str] = {
    0: "Not relevant",
    1: "Weakly or indirectly relevant",
    2: "Relevant",
    3: "Highly relevant",
}
# Levels at or above this count as relevant for binary metrics.
BINARY_RELEVANT_THRESHOLD = 2
NEEDS_REVIEW = "needs_review"
# Accepted review_flag values. "yes"/"no" is the convention used in the
# completed judging file; "yes" and "needs_review" both mark a row for review.
FLAGGED_VALUES = frozenset({NEEDS_REVIEW, "yes"})
REVIEW_FLAG_VALUES = frozenset({"", "no"}) | FLAGGED_VALUES

KEY: tuple[str, str] = ("query_id", "researcher_id")

# Generated from the frozen run; judges must not edit these.
FROZEN_COLUMNS: tuple[str, ...] = (
    "query_id",
    "query_text",
    "researcher_id",
    "researcher_name",
    "system_source",
    "rank",
    "systems",
    "score",
    "evidence_paper_ids",
    "evidence_paper_titles",
    "profile_paper_count",
    "profile_paper_titles",
    "existing_label",
)
EDITABLE_COLUMNS: tuple[str, ...] = (
    "manual_relevance",
    "review_flag",
    "judge_id",
    "judgment_notes",
    "judged_at",
)
COLUMNS: tuple[str, ...] = FROZEN_COLUMNS + EDITABLE_COLUMNS

# Every Phase 1 candidate_pool.csv column and where it lives in the judging file.
PHASE1_COLUMN_MAP: dict[str, str] = {
    "query_id": "query_id",
    "query": "query_text",
    "author_id": "researcher_id",
    "author_name": "researcher_name",
    "in_existing_labels": "existing_label",
    "best_rank": "rank",
    "returned_by": "system_source",
    "systems": "systems",
    "manual_relevance": "manual_relevance",
}

# UTF-8 with a byte-order mark, so Excel shows non-ASCII names correctly.
CSV_ENCODING = "utf-8-sig"


def read_judging_csv(path: Path) -> pd.DataFrame:
    """Read a judging file with every value as text and blanks kept as ``""``."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=CSV_ENCODING)


def write_judging_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a judging file in the spreadsheet-friendly encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding=CSV_ENCODING)
