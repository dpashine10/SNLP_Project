"""Lightweight robustness check over rephrased, ambiguous and Indian-domain queries.

Run from the repository root (needs the Week 4 index)::

    python -m src.week5.scripts.robustness_check

Runs the six frozen systems and the improved Week 4 variant on the queries in
``src.week5.analysis.robustness`` and writes a new folder
``src/week5/artifacts/phase2/robustness/<UTC time>/``:

* ``variant_queries.csv`` / ``variant_summary.csv``: rephrasings of q01, q06 and q07 compared
  with each system's base ranking and with the base query's manual judgments.
* ``system_agreement.csv``: per query, how much the seven top-10 lists overlap.
* ``standalone_top5.csv``: top 5 researchers with evidence for the unlabelled queries.
* ``rankings_evidence.json``: every ranking, in the Phase 1 format.
* ``run.json``: inputs, and whether the base rankings reproduce the frozen run.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.week4.config import CONFIG
from src.week4.evaluation import BASELINE_MODES, EVALUATION_DEPTH, WEEK4_STRATEGIES, build_recommenders
from src.week4.schemas import LabeledQuery
from src.week5.analysis.robustness import (
    STANDALONE_QUERIES,
    VARIANT_QUERIES,
    system_agreement,
    top_researchers,
    variant_rows,
    variant_summary,
)
from src.week5.evaluation.evidence import ranking_record
from src.week5.evaluation.frozen_run import _family
from src.week5.evaluation.variant_evaluation import IMPROVED_STRATEGY
from src.week5.judging.merge import judged_levels, load_frozen_records
from src.week5.judging.schema import CSV_ENCODING
from src.week5.judging.validation import load_valid_judgments
from src.week5.paths import FROZEN_RUN_DIR, JUDGING_TEMPLATE, PHASE2_DIR, active_judgments_file, relative_to_root


def main() -> None:
    """Rank every robustness query with all seven systems and save a dated folder."""
    logging.basicConfig(level=logging.WARNING)
    path = active_judgments_file()
    print(f"Using judgments: {relative_to_root(path)}")
    judgments = judged_levels(load_valid_judgments(path, JUDGING_TEMPLATE))
    frozen = load_frozen_records(FROZEN_RUN_DIR)
    base_ids = sorted({query.base_query_id for query in VARIANT_QUERIES if query.base_query_id})
    base_texts = {record["query_id"]: record["query"] for record in frozen if record["query_id"] in base_ids}
    queries = [(query_id, "original", text) for query_id, text in base_texts.items()]
    queries += [(query.query_id, query.kind, query.text) for query in (*VARIANT_QUERIES, *STANDALONE_QUERIES)]

    recommenders = build_recommenders(
        CONFIG, baseline_modes=BASELINE_MODES, strategies=(*WEEK4_STRATEGIES, IMPROVED_STRATEGY)
    )
    records: list[dict[str, Any]] = []
    for recommender in recommenders:
        for query_id, kind, text in queries:
            ranking = recommender.recommend(text, top_k=EVALUATION_DEPTH)
            record = ranking_record(
                query_id, LabeledQuery(text, frozenset()), recommender.name, _family(recommender), ranking, {}
            )
            records.append({**record, "kind": kind})
    rankings = {
        (record["system"], record["query_id"]): [entry["author_id"] for entry in record["ranking"]]
        for record in records
    }
    frozen_rankings = {
        (record["system"], record["query_id"]): [entry["author_id"] for entry in record["ranking"]]
        for record in frozen
        if record["query_id"] in base_ids
    }
    reproduces_frozen = all(rankings[key] == ranked for key, ranked in frozen_rankings.items())

    rows = variant_rows(rankings, base_texts, judgments)
    summary = variant_summary(rows)
    agreement = system_agreement(rankings, [query_id for query_id, _, _ in queries])
    standalone_ids = {query.query_id for query in STANDALONE_QUERIES}
    top5 = top_researchers([record for record in records if record["query_id"] in standalone_ids])

    out_dir = PHASE2_DIR / "robustness" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(out_dir / "variant_queries.csv", index=False, encoding=CSV_ENCODING)
    pd.DataFrame(summary).to_csv(out_dir / "variant_summary.csv", index=False)
    pd.DataFrame(agreement).to_csv(out_dir / "system_agreement.csv", index=False)
    pd.DataFrame(top5).to_csv(out_dir / "standalone_top5.csv", index=False, encoding=CSV_ENCODING)
    (out_dir / "rankings_evidence.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "judgments_file": relative_to_root(path),
                "frozen_run": relative_to_root(FROZEN_RUN_DIR),
                "systems": [recommender.name for recommender in recommenders],
                "queries": [{"query_id": q, "kind": k, "query": t} for q, k, t in queries],
                "base_rankings_reproduce_frozen_run": reproduces_frozen,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved to {relative_to_root(out_dir)}")
    print(f"Base rankings reproduce the frozen run: {reproduces_frozen}\n")
    print("Rephrased benchmark queries (judgments from the base query; unjudged results are not counted wrong):")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nAgreement between the seven systems (mean pairwise top-10 Jaccard):")
    print(pd.DataFrame(agreement).to_string(index=False))


if __name__ == "__main__":
    main()
