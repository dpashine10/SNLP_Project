"""Evaluate the six frozen rankings with manual judgments only.

Run from the repository root::

    python -m src.week5.scripts.fair_evaluation [--min-judged-fraction 1.0] [--relevant-threshold 2] [--exclude-flagged]

Uses ``manual_relevance`` from the completed judging file (``existing_label``
is never used). ``--exclude-flagged`` treats rows flagged for review as
unjudged, as a sensitivity check; combine it with a lower
``--min-judged-fraction``.

If no query is judged enough, prints a status message and writes nothing.
Otherwise writes a new folder under ``src/week5/artifacts/phase2/fair_evaluation/``.
"""

import argparse
import json
from datetime import UTC, datetime

import pandas as pd

from src.week5.evaluation.fair_evaluation import FairEvaluationSettings, fair_evaluate
from src.week5.judging.merge import judged_levels, load_frozen_records
from src.week5.judging.validation import load_valid_judgments
from src.week5.paths import FROZEN_RUN_DIR, JUDGING_TEMPLATE, PHASE2_DIR, active_judgments_file, relative_to_root


def main() -> None:
    """Run the fair evaluation, or report that it is waiting for labels."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-judged-fraction", type=float, default=1.0)
    parser.add_argument("--relevant-threshold", type=int, default=2)
    parser.add_argument("--exclude-flagged", action="store_true")
    args = parser.parse_args()
    settings = FairEvaluationSettings(args.min_judged_fraction, args.relevant_threshold)

    path = active_judgments_file()
    print(f"Using judgments: {relative_to_root(path)}" + (" (flagged rows excluded)" if args.exclude_flagged else ""))
    judgments = judged_levels(load_valid_judgments(path, JUDGING_TEMPLATE), exclude_flagged=args.exclude_flagged)
    result = fair_evaluate(load_frozen_records(FROZEN_RUN_DIR), judgments, settings)
    print(
        f"Judged pairs: {result.judged_pairs} of {result.total_pairs} "
        f"(unjudged: {result.total_pairs - result.judged_pairs}). Unjudged pairs are excluded, never counted as irrelevant."
    )
    for query_id, (judged, total) in result.coverage_by_query.items():
        status = "evaluated" if query_id in result.evaluated_queries else f"excluded: {result.excluded_queries[query_id]}"
        print(f"  {query_id}: {judged}/{total} judged -> {status}")
    if not result.ready:
        print("\nSTATUS: waiting for manual labels. No fair evaluation was produced.")
        return

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = PHASE2_DIR / "fair_evaluation" / (f"{stamp}-exclude-flagged" if args.exclude_flagged else stamp)
    out_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(result.summary).to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(result.per_query).to_csv(out_dir / "per_query.csv", index=False)
    (out_dir / "coverage.json").write_text(
        json.dumps(
            {
                "judgments_file": relative_to_root(path),
                "label_column": "manual_relevance",
                "exclude_flagged": args.exclude_flagged,
                "settings": vars(settings),
                "judged_pairs": result.judged_pairs,
                "total_pairs": result.total_pairs,
                "evaluated_queries": result.evaluated_queries,
                "excluded_queries": result.excluded_queries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nFair evaluation on {len(result.evaluated_queries)} queries saved to {relative_to_root(out_dir)}\n")
    print(pd.DataFrame(result.summary).to_string(index=False))


if __name__ == "__main__":
    main()
