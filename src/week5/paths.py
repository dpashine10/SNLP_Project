"""Locations of Week 5 artifacts, resolved from this file so commands work from any directory."""

from pathlib import Path

from src.week4.config import PROJECT_ROOT

WEEK5_DIR: Path = Path(__file__).resolve().parent
ARTIFACTS_DIR: Path = WEEK5_DIR / "artifacts"
RUNS_DIR: Path = ARTIFACTS_DIR / "runs"
REPRODUCIBILITY_DIR: Path = ARTIFACTS_DIR / "reproducibility"

# Phase 2 reads the first frozen Phase 1 run (byte-identical to the second).
FROZEN_RUN_DIR: Path = RUNS_DIR / "20261007T154203Z"
PHASE2_DIR: Path = ARTIFACTS_DIR / "phase2"
JUDGING_DIR: Path = WEEK5_DIR / "judging"
JUDGING_TEMPLATE: Path = JUDGING_DIR / "templates" / "judging_template.csv"
JUDGMENTS_FILE: Path = JUDGING_DIR / "manual_judgments.csv"
COMPLETED_JUDGMENTS_FILE: Path = JUDGING_DIR / "manual_judgments_completed.csv"


def active_judgments_file() -> Path:
    """The judging file the scripts use: the completed file once it exists, else the blank one."""
    return COMPLETED_JUDGMENTS_FILE if COMPLETED_JUDGMENTS_FILE.is_file() else JUDGMENTS_FILE


def relative_to_root(path: Path) -> str:
    """Return ``path`` relative to the repository root, so artifacts hold no absolute paths."""
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return resolved.as_posix()
