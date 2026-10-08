"""Rankings for the improved Week 4 variant, in the same format as the frozen records.

The variant runs through Week 4's own pipeline (``build_week4_pipeline``), on
the same benchmark queries and query IDs as the frozen Phase 1 run. Only the
ranking strategy differs.
"""

import dataclasses
from collections.abc import Sequence
from typing import Any

from src.week4.config import RankingStrategyName, Week4Config
from src.week4.evaluation import EVALUATION_DEPTH, METRICS
from src.week4.schemas import LabeledQuery
from src.week4.week4_pipeline import build_week4_pipeline
from src.week5.evaluation.evidence import ranking_record

IMPROVED_STRATEGY: RankingStrategyName = "top_k_average_author_discount"


def variant_records(
    config: Week4Config, queries: Sequence[LabeledQuery], strategy: RankingStrategyName = IMPROVED_STRATEGY
) -> list[dict[str, Any]]:
    """Rank every usable benchmark query with ``strategy``; query IDs match the frozen run."""
    pipeline = build_week4_pipeline(dataclasses.replace(config, ranking_strategy=strategy))
    usable = [query for query in queries if query.relevant_author_ids]
    records = []
    for index, labeled_query in enumerate(usable, start=1):
        ranking = pipeline.recommend(labeled_query.query, top_k=EVALUATION_DEPTH)
        ranked_ids = tuple(result.author.author_id for result in ranking)
        old_label_metrics = {
            name: metric(ranked_ids, labeled_query.relevant_author_ids) for name, metric in METRICS.items()
        }
        records.append(
            ranking_record(f"q{index:02d}", labeled_query, pipeline.name, "week4", ranking, old_label_metrics)
        )
    return records


def unjudged_pairs(records: Sequence[dict[str, Any]], judged: set[tuple[str, str]]) -> list[dict[str, Any]]:
    """Researchers in ``records`` that have no judgment yet, for a later judging addendum."""
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        for entry in record["ranking"]:
            key = (record["query_id"], entry["author_id"])
            if key not in judged and key not in seen:
                seen[key] = {
                    "query_id": record["query_id"],
                    "query_text": record["query"],
                    "researcher_id": entry["author_id"],
                    "researcher_name": entry["author_name"] or "",
                    "system": record["system"],
                    "rank": entry["rank"],
                    "evidence_paper_titles": " || ".join(paper["title"] or "" for paper in entry["evidence"]),
                    "manual_relevance": "",
                }
    return list(seen.values())
