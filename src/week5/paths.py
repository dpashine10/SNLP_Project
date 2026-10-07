"""Locations of Week 5 artifacts, resolved from this file so commands work from any directory."""

from pathlib import Path

from src.week4.config import PROJECT_ROOT

WEEK5_DIR: Path = Path(__file__).resolve().parent
ARTIFACTS_DIR: Path = WEEK5_DIR / "artifacts"
RUNS_DIR: Path = ARTIFACTS_DIR / "runs"
REPRODUCIBILITY_DIR: Path = ARTIFACTS_DIR / "reproducibility"


def relative_to_root(path: Path) -> str:
    """Return ``path`` relative to the repository root, so artifacts hold no absolute paths."""
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return resolved.as_posix()
