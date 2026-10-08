"""Create the frozen judging template and the editable judging file.

Run from the repository root::

    python -m src.week5.scripts.prepare_judging

The template is regenerated from the frozen Phase 1 run and must match any
existing template exactly. ``manual_judgments.csv`` is created only if it does
not exist yet, so filled-in judgments are never overwritten.
"""

import logging

from src.week5.judging.schema import read_judging_csv, write_judging_csv
from src.week5.judging.template import build_template
from src.week5.paths import FROZEN_RUN_DIR, JUDGING_TEMPLATE, JUDGMENTS_FILE, relative_to_root


def main() -> None:
    """Write the template (if absent or identical) and the editable file (if absent)."""
    logging.basicConfig(level=logging.WARNING)
    template = build_template(FROZEN_RUN_DIR)
    if JUDGING_TEMPLATE.exists():
        if not read_judging_csv(JUDGING_TEMPLATE).equals(template):
            raise SystemExit(
                f"{relative_to_root(JUDGING_TEMPLATE)} differs from the frozen run; "
                "the template is a fixed reference and is not overwritten."
            )
        print(f"Template unchanged: {relative_to_root(JUDGING_TEMPLATE)}")
    else:
        write_judging_csv(template, JUDGING_TEMPLATE)
        print(f"Wrote template: {relative_to_root(JUDGING_TEMPLATE)} ({len(template)} pairs)")
    if JUDGMENTS_FILE.exists():
        print(f"Kept existing judging file (never overwritten): {relative_to_root(JUDGMENTS_FILE)}")
    else:
        write_judging_csv(template, JUDGMENTS_FILE)
        print(f"Wrote judging file to fill in: {relative_to_root(JUDGMENTS_FILE)}")


if __name__ == "__main__":
    main()
