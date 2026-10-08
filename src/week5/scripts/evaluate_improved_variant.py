"""Evaluate the improved Week 4 variant alongside the six frozen systems.

Run from the repository root (needs the Week 4 index)::

    python -m src.week5.scripts.evaluate_improved_variant

Writes a new folder ``src/week5/artifacts/phase2/improved_variant/<UTC time>/``:

* ``fair_summary.csv`` / ``fair_per_query.csv``: all seven systems on ``manual_relevance``.
  Query eligibility comes from the frozen judged pool. Variant results outside that
  pool are removed from its ranking (``unjudged_removed``), never counted as wrong.
* ``old_label_summary.csv``: the same seven systems on the old TF-IDF-derived labels.
* ``coauthor_concentration.csv``: distinct papers and shared-paper groups per Week 4 top 10.
* ``variant_rankings_evidence.json``: the variant's rankings, in the Phase 1 format.
* ``unjudged_variant_pairs.csv``: variant results that still need a judgment.
* ``run.json``: inputs and coverage.
"""

import json
import logging
from datetime import UTC, datetime

import pandas as pd

from src.week4.config import CONFIG
from src.week4.evaluation import QueryResult, load_labeled_queries, summarize
from src.week5.analysis.error_analysis import coauthor_tie_report
from src.week5.evaluation.fair_evaluation import fair_evaluate
from src.week5.evaluation.variant_evaluation import IMPROVED_STRATEGY, unjudged_pairs, variant_records
from src.week5.judging.merge import judged_levels, load_frozen_records
from src.week5.judging.schema import CSV_ENCODING
from src.week5.judging.validation import load_valid_judgments
from src.week5.paths import FROZEN_RUN_DIR, JUDGING_TEMPLATE, PHASE2_DIR, active_judgments_file, relative_to_root


def main() -> None:
    """Rank with the variant, evaluate all seven systems and save a dated folder."""
    logging.basicConfig(level=logging.WARNING)
    path = active_judgments_file()
    print(f"Using judgments: {relative_to_root(path)}")
    levels = judged_levels(load_valid_judgments(path, JUDGING_TEMPLATE))
    frozen = load_frozen_records(FROZEN_RUN_DIR)
    variant = variant_records(CONFIG, load_labeled_queries(CONFIG))
    records = frozen + variant

    fair = fair_evaluate(records, levels, pool_records=frozen)
    old = summarize(
        [
            QueryResult(
                system=record["system"],
                query=record["query"],
                ranked_author_ids=tuple(entry["author_id"] for entry in record["ranking"]),
                metrics=record["metrics"],
            )
            for record in records
        ]
    )
    concentration = pd.DataFrame(coauthor_tie_report(records)).groupby("system", sort=False).agg(
        mean_distinct_best_papers=("distinct_best_papers", "mean"),
        mean_largest_shared_paper_group=("largest_shared_paper_group", "mean"),
        rankings_with_shared_paper_flag=("flag_shared_paper", "sum"),
    ).round(3)
    missing = unjudged_pairs(variant, set(levels))

    out_dir = PHASE2_DIR / "improved_variant" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(fair.summary).to_csv(out_dir / "fair_summary.csv", index=False)
    pd.DataFrame(fair.per_query).to_csv(out_dir / "fair_per_query.csv", index=False)
    pd.DataFrame(old).to_csv(out_dir / "old_label_summary.csv", index=False)
    concentration.to_csv(out_dir / "coauthor_concentration.csv")
    pd.DataFrame(
        missing, columns=["query_id", "query_text", "researcher_id", "researcher_name", "system", "rank",
                          "evidence_paper_titles", "manual_relevance"]
    ).to_csv(out_dir / "unjudged_variant_pairs.csv", index=False, encoding=CSV_ENCODING)
    (out_dir / "variant_rankings_evidence.json").write_text(
        json.dumps(variant, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "variant": IMPROVED_STRATEGY,
                "judgments_file": relative_to_root(path),
                "frozen_run": relative_to_root(FROZEN_RUN_DIR),
                "evaluated_queries": fair.evaluated_queries,
                "excluded_queries": fair.excluded_queries,
                "variant_results": sum(len(record["ranking"]) for record in variant),
                "variant_results_unjudged": sum(row["unjudged_removed"] for row in fair.per_query
                                                if row["system"].endswith(f"({IMPROVED_STRATEGY})")),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved to {relative_to_root(out_dir)}\n")
    print("Manual labels (fair evaluation):")
    print(pd.DataFrame(fair.summary).to_string(index=False))
    print("\nOld TF-IDF-derived labels:")
    print(pd.DataFrame(old).to_string(index=False))
    print("\nWeek 4 co-author concentration (per top 10, averaged over queries):")
    print(concentration.to_string())
    print(f"\nVariant results not in the judged pool: {len(missing)} (listed in unjudged_variant_pairs.csv)")


if __name__ == "__main__":
    main()
