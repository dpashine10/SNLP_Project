"""Validate the manual judging file against the frozen template.

Run from the repository root::

    python -m src.week5.scripts.validate_judgments [PATH]

Exits with status 1 if any error is found.
"""

import sys
from pathlib import Path

from src.week5.judging.validation import validate_judgments
from src.week5.paths import JUDGING_TEMPLATE, active_judgments_file, relative_to_root


def main(argv: list[str]) -> int:
    """Print the validation summary and every error; return the exit status."""
    path = Path(argv[0]) if argv else active_judgments_file()
    report = validate_judgments(path, JUDGING_TEMPLATE)
    print(f"File: {relative_to_root(path)}")
    print(
        f"Pairs: {report.total} | judged: {report.judged} | unjudged: {report.unjudged} "
        f"| flagged for review: {report.flagged_for_review}"
    )
    if report.by_level:
        print("Judged by level: " + ", ".join(f"{level}: {count}" for level, count in sorted(report.by_level.items())))
    if report.ok:
        print("Valid.")
        return 0
    print(f"{len(report.errors)} error(s):")
    for error in report.errors:
        print(f"  - {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
