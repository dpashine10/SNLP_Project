"""Validates a filled-in judging file against the frozen template.

Blank ``manual_relevance`` means unjudged and is always valid; it is never
read as "not relevant".
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.week5.judging.schema import (
    COLUMNS,
    FLAGGED_VALUES,
    FROZEN_COLUMNS,
    KEY,
    RELEVANCE_SCALE,
    REVIEW_FLAG_VALUES,
    read_judging_csv,
)

_VALID_LEVELS = {str(level) for level in RELEVANCE_SCALE}
_MAX_LISTED = 20


@dataclass
class ValidationReport:
    """Outcome of validating a judging file."""

    errors: list[str] = field(default_factory=list)
    total: int = 0
    judged: int = 0
    flagged_for_review: int = 0
    by_level: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def unjudged(self) -> int:
        return self.total - self.judged


def validate_judgments(judgments_path: Path, template_path: Path) -> ValidationReport:
    """Check a judging file: columns, values, required fields, pairs and frozen fields.

    Raises:
        FileNotFoundError: If either file does not exist.
    """
    for path in (judgments_path, template_path):
        if not path.is_file():
            raise FileNotFoundError(f"Judging file not found: {path}")
    return validate_frame(read_judging_csv(judgments_path), read_judging_csv(template_path))


def load_valid_judgments(judgments_path: Path, template_path: Path) -> pd.DataFrame:
    """Read a judging file, refusing to continue if it does not validate.

    Raises:
        ValueError: If validation finds any error (all errors are listed).
    """
    report = validate_judgments(judgments_path, template_path)
    if not report.ok:
        raise ValueError(
            "The judging file does not validate; run `python -m src.week5.scripts.validate_judgments`.\n"
            + "\n".join(report.errors)
        )
    return read_judging_csv(judgments_path)


def validate_frame(judgments: pd.DataFrame, template: pd.DataFrame) -> ValidationReport:
    """Validate an in-memory judging table against the template table."""
    report = ValidationReport(total=len(judgments))
    missing = [column for column in COLUMNS if column not in judgments.columns]
    if missing:
        report.errors.append(f"Missing required columns: {', '.join(missing)}.")
        return report
    _check_duplicates(judgments, report)
    _check_pairs(judgments, template, report)
    _check_frozen_fields(judgments, template, report)
    _check_rows(judgments, report)
    return report


def _row(index: object) -> int:
    """Spreadsheet row number of a data row (row 1 is the header)."""
    return int(str(index)) + 2


def _check_duplicates(judgments: pd.DataFrame, report: ValidationReport) -> None:
    duplicated = judgments[judgments.duplicated(list(KEY), keep=False)]
    for (query_id, researcher_id), group in duplicated.groupby(list(KEY)):
        rows = ", ".join(str(_row(index)) for index in group.index)
        report.errors.append(f"Duplicate pair {query_id} / {researcher_id} on rows {rows}.")


def _check_pairs(judgments: pd.DataFrame, template: pd.DataFrame, report: ValidationReport) -> None:
    found = set(zip(judgments["query_id"], judgments["researcher_id"], strict=True))
    expected = set(zip(template["query_id"], template["researcher_id"], strict=True))
    for query_id, researcher_id in sorted(expected - found)[:_MAX_LISTED]:
        report.errors.append(f"Missing pair from the frozen evidence: {query_id} / {researcher_id}.")
    for query_id, researcher_id in sorted(found - expected)[:_MAX_LISTED]:
        report.errors.append(f"Pair not in the frozen evidence: {query_id} / {researcher_id}.")


def _check_frozen_fields(judgments: pd.DataFrame, template: pd.DataFrame, report: ValidationReport) -> None:
    expected = template.drop_duplicates(list(KEY)).set_index(list(KEY))
    listed = 0
    for index, row in judgments.iterrows():
        key = (row["query_id"], row["researcher_id"])
        if key not in expected.index:
            continue
        for column in FROZEN_COLUMNS:
            if column in KEY or row[column] == expected.at[key, column]:
                continue
            listed += 1
            if listed <= _MAX_LISTED:
                report.errors.append(
                    f"Row {_row(index)}: frozen column '{column}' was changed "
                    f"(expected {expected.at[key, column]!r}, found {row[column]!r})."
                )
    if listed > _MAX_LISTED:
        report.errors.append(f"... and {listed - _MAX_LISTED} more changed frozen values.")


def _check_rows(judgments: pd.DataFrame, report: ValidationReport) -> None:
    for index, row in judgments.iterrows():
        relevance = row["manual_relevance"].strip()
        flag = row["review_flag"].strip()
        where = f"Row {_row(index)} ({row['query_id']} / {row['researcher_name'] or row['researcher_id']})"
        if relevance and relevance not in _VALID_LEVELS:
            report.errors.append(f"{where}: manual_relevance must be blank, 0, 1, 2 or 3, found {relevance!r}.")
            continue
        if flag not in REVIEW_FLAG_VALUES:
            allowed = ", ".join(repr(value) for value in sorted(REVIEW_FLAG_VALUES))
            report.errors.append(f"{where}: review_flag must be one of {allowed}, found {flag!r}.")
        if flag in FLAGGED_VALUES:
            # A flagged row may be unjudged or carry a provisional level; either way it needs a reason.
            report.flagged_for_review += 1
            if not row["judgment_notes"].strip():
                report.errors.append(f"{where}: explain in judgment_notes why the row needs review.")
        if not relevance:
            continue
        report.judged += 1
        report.by_level[relevance] = report.by_level.get(relevance, 0) + 1
        for column in ("judge_id", "judgment_notes", "judged_at"):
            if not row[column].strip():
                report.errors.append(f"{where}: judged rows need {column}.")
        if row["judged_at"].strip() and not _is_iso_datetime(row["judged_at"].strip()):
            report.errors.append(f"{where}: judged_at must be an ISO date such as 2026-10-08, found {row['judged_at']!r}.")


def _is_iso_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True
