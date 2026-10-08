"""Summary of the pooled candidates and how far manual judging has progressed.

Unjudged pairs are reported as unjudged, never as negative examples.
"""

from collections import Counter
from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.week5.judging.schema import FLAGGED_VALUES, RELEVANCE_SCALE


def summarize_pool(judgments: pd.DataFrame, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count pairs, judgments, systems, queries, researchers and evidence papers."""
    judged = judgments["manual_relevance"].str.strip() != ""
    systems: dict[str, set[tuple[str, str]]] = {}
    evidence_papers: set[str] = set()
    for record in records:
        pairs = systems.setdefault(record["system"], set())
        for entry in record["ranking"]:
            pairs.add((record["query_id"], entry["author_id"]))
            evidence_papers.update(paper["paper_id"] for paper in entry["evidence"])
    judged_pairs = set(zip(judgments.loc[judged, "query_id"], judgments.loc[judged, "researcher_id"], strict=True))
    levels = Counter(judgments.loc[judged, "manual_relevance"].str.strip())
    by_query = judgments.assign(judged=judged).groupby("query_id").agg(
        query=("query_text", "first"),
        pairs=("researcher_id", "size"),
        judged=("judged", "sum"),
        in_existing_labels=("existing_label", lambda labels: int((labels == "1").sum())),
    )
    return {
        "total_pairs": len(judgments),
        "judged_pairs": int(judged.sum()),
        "unjudged_pairs": int((~judged).sum()),
        "flagged_for_review": int(judgments["review_flag"].str.strip().isin(FLAGGED_VALUES).sum()),
        "pairs_in_existing_labels": int((judgments["existing_label"] == "1").sum()),
        "unique_queries": int(judgments["query_id"].nunique()),
        "unique_researchers": int(judgments["researcher_id"].nunique()),
        "distinct_week4_evidence_papers": len(evidence_papers),
        "by_system": {
            system: {"pairs": len(pairs), "judged": len(pairs & judged_pairs)} for system, pairs in systems.items()
        },
        "by_query": {
            query_id: {key: (int(value) if key != "query" else value) for key, value in row.items()}
            for query_id, row in by_query.to_dict("index").items()
        },
        "by_relevance_level": {
            f"{level} ({label})": levels.get(str(level), 0) for level, label in RELEVANCE_SCALE.items()
        }
        | {"unjudged": int((~judged).sum())},
    }


def summary_markdown(summary: dict[str, Any]) -> str:
    """Render the summary as a short Markdown report."""
    lines = [
        "# Candidate pool summary (Week 5, Phase 2)",
        "",
        "Unjudged pairs are unjudged, not negative examples.",
        "",
        "| Measure | Count |",
        "|---|---|",
    ]
    for key in (
        "total_pairs", "judged_pairs", "unjudged_pairs", "flagged_for_review", "pairs_in_existing_labels",
        "unique_queries", "unique_researchers", "distinct_week4_evidence_papers",
    ):
        lines.append(f"| {key.replace('_', ' ')} | {summary[key]} |")
    lines += ["", "## By system", "", "| System | Pooled pairs | Judged |", "|---|---|---|"]
    lines += [f"| {system} | {row['pairs']} | {row['judged']} |" for system, row in summary["by_system"].items()]
    lines += ["", "## By query", "", "| Query | Text | Pairs | Judged | In existing labels |", "|---|---|---|---|---|"]
    lines += [
        f"| {query_id} | {row['query']} | {row['pairs']} | {row['judged']} | {row['in_existing_labels']} |"
        for query_id, row in summary["by_query"].items()
    ]
    lines += ["", "## By manual relevance level", "", "| Level | Pairs |", "|---|---|"]
    lines += [f"| {level} | {count} |" for level, count in summary["by_relevance_level"].items()]
    return "\n".join(lines) + "\n"
