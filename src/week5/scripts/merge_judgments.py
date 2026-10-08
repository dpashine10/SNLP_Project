"""Merge validated manual judgments into the frozen rankings.

Run from the repository root::

    python -m src.week5.scripts.merge_judgments

Writes ``src/week5/artifacts/phase2/judged_rankings.json`` (regenerated on
each merge; the frozen Phase 1 files are only read).
"""

import json

from src.week5.judging.merge import load_frozen_records, merge_judgments
from src.week5.judging.validation import load_valid_judgments
from src.week5.paths import FROZEN_RUN_DIR, JUDGING_TEMPLATE, PHASE2_DIR, active_judgments_file, relative_to_root

OUTPUT = PHASE2_DIR / "judged_rankings.json"


def main() -> None:
    """Validate, merge and save; print how many ranked entries carry a judgment."""
    print(f"Using judgments: {relative_to_root(active_judgments_file())}")
    judgments = load_valid_judgments(active_judgments_file(), JUDGING_TEMPLATE)
    merged = merge_judgments(load_frozen_records(FROZEN_RUN_DIR), judgments)
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    entries = [entry for record in merged for entry in record["ranking"]]
    judged = sum(entry["manual_relevance"] is not None for entry in entries)
    print(f"Merged {judged} judged of {len(entries)} ranked entries into {relative_to_root(OUTPUT)}")


if __name__ == "__main__":
    main()
