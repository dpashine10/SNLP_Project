"""Structured error analysis of the frozen rankings.

Run from the repository root::

    python -m src.week5.scripts.analyze_errors

Writes to ``src/week5/artifacts/phase2/``:

* ``coauthor_ties.csv``: concentration of each Week 4 top 10 on few papers and tied scores.
* ``error_cases.csv``: cases confirmed directly by the frozen evidence.
* ``error_review_queue.csv``: top-ranked researchers whose manual relevance decides further errors.
* ``error_analysis.md``: a short summary.
"""

from collections import Counter

import pandas as pd

from src.week5.analysis.error_analysis import (
    ERROR_CASE_COLUMNS,
    FEW_PAPERS_MAX,
    REVIEW_DEPTH,
    SHARED_PAPER_MIN,
    TIE_GROUP_MIN,
    coauthor_tie_report,
    confirmed_cases,
    review_queue,
)
from src.week5.judging.merge import judged_levels, load_frozen_records
from src.week5.judging.schema import CSV_ENCODING
from src.week5.judging.validation import load_valid_judgments
from src.week5.paths import FROZEN_RUN_DIR, JUDGING_TEMPLATE, PHASE2_DIR, active_judgments_file, relative_to_root


def main() -> None:
    """Build the three tables and the summary from the frozen rankings and current judgments."""
    print(f"Using judgments: {relative_to_root(active_judgments_file())}")
    records = load_frozen_records(FROZEN_RUN_DIR)
    judgments = judged_levels(load_valid_judgments(active_judgments_file(), JUDGING_TEMPLATE))
    ties = coauthor_tie_report(records)
    cases = confirmed_cases(records, judgments)
    queue = review_queue(records, judgments)

    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ties).to_csv(PHASE2_DIR / "coauthor_ties.csv", index=False, encoding=CSV_ENCODING)
    pd.DataFrame(cases, columns=list(ERROR_CASE_COLUMNS)).to_csv(
        PHASE2_DIR / "error_cases.csv", index=False, encoding=CSV_ENCODING
    )
    pd.DataFrame(queue).to_csv(PHASE2_DIR / "error_review_queue.csv", index=False, encoding=CSV_ENCODING)
    markdown = _markdown(ties, cases, queue)
    (PHASE2_DIR / "error_analysis.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved to {relative_to_root(PHASE2_DIR)}/")


def _markdown(ties: list[dict[str, object]], cases: list[dict[str, object]], queue: list[dict[str, object]]) -> str:
    flagged = [row for row in ties if row["flag_shared_paper"] or row["flag_identical_scores"] or row["flag_few_papers"]]
    categories = Counter(str(case["error_category"]) for case in cases)
    pending = sum(1 for row in queue if row["manual_relevance"] == "")
    lines = [
        "# Error analysis (Week 5, Phase 2)",
        "",
        "Generated from the frozen Phase 1 rankings by `python -m src.week5.scripts.analyze_errors`.",
        "",
        "## Week 4 co-author concentration",
        "",
        f"Flags: a best-evidence paper shared by at least {SHARED_PAPER_MIN} top-10 researchers; at least "
        f"{TIE_GROUP_MIN} identical scores; or a top 10 drawn from {FEW_PAPERS_MAX} or fewer distinct papers.",
        "",
        "| System | Query | Distinct best papers | Largest shared-paper group | Largest identical-score group |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['system']} | {row['query_id']} | {row['distinct_best_papers']} | "
        f"{row['largest_shared_paper_group']} | {row['largest_identical_score_group']} |"
        for row in ties
    ]
    lines += [
        "",
        f"{len(flagged)} of {len(ties)} Week 4 system–query rankings trigger at least one flag.",
        "",
        "## Confirmed error cases",
        "",
        f"{len(cases)} cases are confirmed directly by the evidence (no relevance assumption): "
        + (", ".join(f"{count} × {category}" for category, count in categories.items()) or "none")
        + ". See `error_cases.csv`.",
        "",
        "## Review queue",
        "",
        f"{len(queue)} top-{REVIEW_DEPTH} researchers across Week 3 semantic and the Week 4 strategies; {pending} still need a "
        "manual judgment before relevance-dependent errors (such as generic semantic similarity or a weak "
        "evidence paper) can be categorised. See `error_review_queue.csv`.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
