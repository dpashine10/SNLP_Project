"""Combines validated manual judgments with the frozen Phase 1 rankings.

Rankings are never changed. Each ranked researcher only gains its manual
judgment; unjudged researchers get ``None``, never 0.
"""

import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.week5.judging.schema import FLAGGED_VALUES

Judgments = dict[tuple[str, str], int]


def judged_levels(judgments: pd.DataFrame, *, exclude_flagged: bool = False) -> Judgments:
    """Map each judged (query_id, researcher_id) pair to its 0–3 level; unjudged pairs are absent.

    With ``exclude_flagged``, rows flagged for review are also left out (treated
    as unjudged, never as irrelevant), for a sensitivity check.
    """
    return {
        (row.query_id, row.researcher_id): int(row.manual_relevance)
        for row in judgments.itertuples(index=False)
        if str(row.manual_relevance).strip()
        and not (exclude_flagged and row.review_flag.strip() in FLAGGED_VALUES)
    }


def load_frozen_records(run_dir: Path) -> list[dict[str, Any]]:
    """Read a frozen run's rankings evidence (read-only)."""
    path = run_dir / "rankings_evidence.json"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen Phase 1 artifact not found: {path}")
    records: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return records


def merge_judgments(records: Sequence[dict[str, Any]], judgments: pd.DataFrame) -> list[dict[str, Any]]:
    """Return copies of the frozen records with each ranked researcher's manual judgment attached."""
    levels = judged_levels(judgments)
    flagged = {
        (row.query_id, row.researcher_id)
        for row in judgments.itertuples(index=False)
        if row.review_flag.strip() in FLAGGED_VALUES
    }
    merged = copy.deepcopy(list(records))
    for record in merged:
        for entry in record["ranking"]:
            key = (record["query_id"], entry["author_id"])
            entry["manual_relevance"] = levels.get(key)
            entry["needs_review"] = key in flagged
    return merged
