"""Compare the old TF-IDF-derived labels with the new manual relevance labels.

Run from the repository root::

    python -m src.week5.scripts.compare_labels

Writes ``label_agreement.{json,md}`` and ``label_disagreements.csv`` to
``src/week5/artifacts/phase2/``. Requires a fully judged file.
"""

import json

from src.week5.analysis.label_agreement import agreement_report, disagreements, report_markdown
from src.week5.judging.schema import CSV_ENCODING
from src.week5.judging.validation import load_valid_judgments
from src.week5.paths import JUDGING_TEMPLATE, PHASE2_DIR, active_judgments_file, relative_to_root


def main() -> None:
    """Build the agreement report from the validated judging file."""
    path = active_judgments_file()
    print(f"Using judgments: {relative_to_root(path)}")
    judgments = load_valid_judgments(path, JUDGING_TEMPLATE)
    unjudged = int((judgments["manual_relevance"].str.strip() == "").sum())
    if unjudged:
        raise SystemExit(f"{unjudged} pairs are still unjudged; label agreement needs every pair judged.")
    report = agreement_report(judgments)
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    (PHASE2_DIR / "label_agreement.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = report_markdown(report)
    (PHASE2_DIR / "label_agreement.md").write_text(markdown, encoding="utf-8")
    disagreements(judgments).to_csv(PHASE2_DIR / "label_disagreements.csv", index=False, encoding=CSV_ENCODING)
    print(markdown)
    print(f"Saved to {relative_to_root(PHASE2_DIR)}/label_agreement.{{json,md}} and label_disagreements.csv")


if __name__ == "__main__":
    main()
