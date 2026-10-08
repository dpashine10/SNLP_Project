"""Summarise the candidate pool and judging progress.

Run from the repository root::

    python -m src.week5.scripts.summarize_pool

Writes ``pool_summary.json`` and ``pool_summary.md`` to ``src/week5/artifacts/phase2/``.
"""

import json

from src.week5.analysis.pool_summary import summarize_pool, summary_markdown
from src.week5.judging.merge import load_frozen_records
from src.week5.judging.validation import load_valid_judgments
from src.week5.paths import FROZEN_RUN_DIR, JUDGING_TEMPLATE, PHASE2_DIR, active_judgments_file, relative_to_root


def main() -> None:
    """Summarise the current judging file against the frozen rankings."""
    print(f"Using judgments: {relative_to_root(active_judgments_file())}")
    judgments = load_valid_judgments(active_judgments_file(), JUDGING_TEMPLATE)
    summary = summarize_pool(judgments, load_frozen_records(FROZEN_RUN_DIR))
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    (PHASE2_DIR / "pool_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown = summary_markdown(summary)
    (PHASE2_DIR / "pool_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved to {relative_to_root(PHASE2_DIR)}/pool_summary.{{json,md}}")


if __name__ == "__main__":
    main()
