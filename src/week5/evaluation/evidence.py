"""Evidence records for later error analysis, and the candidate pool for manual labelling.

Records only describe what each system returned; nothing here changes a
ranking, a score or a benchmark label.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from src.week4.schemas import LabeledQuery, PaperHit, ResearcherScore


def ranking_record(
    query_id: str,
    labeled_query: LabeledQuery,
    system: str,
    family: str,
    ranking: Sequence[ResearcherScore],
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    """Describe one system's ranking for one query, with the labels currently used to score it."""
    relevant = labeled_query.relevant_author_ids
    return {
        "query_id": query_id,
        "query": labeled_query.query,
        "system": system,
        "family": family,
        "relevant_author_ids": sorted(relevant),
        "metrics": dict(metrics),
        "ranking": [
            _researcher_entry(rank, result, relevant)
            for rank, result in enumerate(ranking, start=1)
        ],
    }


def _researcher_entry(
    rank: int, result: ResearcherScore, relevant: frozenset[str]
) -> dict[str, Any]:
    has_evidence = bool(result.evidence)
    return {
        "rank": rank,
        "author_id": result.author.author_id,
        "author_name": result.author.name or None,
        "score": result.score,
        "labelled_relevant": result.author.author_id in relevant,
        # Week 3 exposes no paper-level evidence, so these stay empty for it.
        "matched_papers": result.matched_papers if has_evidence else None,
        "evidence": [_paper_entry(hit) for hit in result.evidence],
    }


def _paper_entry(hit: PaperHit) -> dict[str, Any]:
    return {
        "paper_id": hit.paper.paper_id,
        "title": hit.paper.title or None,
        "publication_year": hit.paper.metadata.get("publication_year"),
        "score": hit.score,
    }


def candidate_pool(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pool every system's returned researchers per query for later manual judging.

    ``manual_relevance`` is left blank: labels are never generated
    automatically. ``in_existing_labels`` only shows whether the researcher is
    in the current (TF-IDF-derived) relevant set.
    """
    pool: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        for entry in record["ranking"]:
            key = (record["query_id"], entry["author_id"])
            candidate = pool.setdefault(
                key,
                {
                    "query_id": record["query_id"],
                    "query": record["query"],
                    "author_id": entry["author_id"],
                    "author_name": entry["author_name"],
                    "in_existing_labels": entry["labelled_relevant"],
                    "best_rank": entry["rank"],
                    "returned_by": [],
                },
            )
            candidate["best_rank"] = min(candidate["best_rank"], entry["rank"])
            candidate["returned_by"].append(f"{record['system']} #{entry['rank']}")
    rows = [
        {
            **candidate,
            "systems": len(candidate["returned_by"]),
            "returned_by": "; ".join(candidate["returned_by"]),
            "manual_relevance": "",
        }
        for candidate in pool.values()
    ]
    rows.sort(key=lambda row: (row["query_id"], -row["systems"], row["best_rank"], row["author_id"]))
    return rows
