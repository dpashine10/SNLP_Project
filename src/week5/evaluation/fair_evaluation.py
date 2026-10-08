"""Fair evaluation of the six frozen rankings using manual judgments only.

Rankings are the frozen Phase 1 rankings and are never changed; only the
relevance labels differ from the old benchmark. Unjudged researchers are never
treated as irrelevant:

* A query is evaluated only when at least ``min_judged_fraction`` of its
  pooled pairs are judged (default 1.0, i.e. fully judged).
* Within an evaluated query, unjudged researchers are removed from each
  ranking before scoring (condensed lists). With full judging nothing is
  removed, because the pool contains every system's top 10.
* Levels at or above ``relevant_threshold`` (default 2) count as relevant for
  the binary metrics; ``ndcg@10_graded`` uses the 0–3 levels as gains.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.week4.evaluation import METRICS
from src.week5.judging.merge import Judgments
from src.week5.judging.schema import BINARY_RELEVANT_THRESHOLD

_DEPTH = 10


@dataclass(frozen=True)
class FairEvaluationSettings:
    """Which judged queries to use and how levels map to relevance."""

    min_judged_fraction: float = 1.0
    relevant_threshold: int = BINARY_RELEVANT_THRESHOLD

    def __post_init__(self) -> None:
        if not 0.0 < self.min_judged_fraction <= 1.0:
            raise ValueError("min_judged_fraction must be in (0, 1].")
        if self.relevant_threshold not in (1, 2, 3):
            raise ValueError("relevant_threshold must be 1, 2 or 3.")


@dataclass
class FairEvaluationResult:
    """Coverage, per-query metrics and per-system means of a fair evaluation."""

    total_pairs: int
    judged_pairs: int
    coverage_by_query: dict[str, tuple[int, int]]
    evaluated_queries: list[str] = field(default_factory=list)
    excluded_queries: dict[str, str] = field(default_factory=dict)
    per_query: list[dict[str, Any]] = field(default_factory=list)
    summary: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return bool(self.evaluated_queries)


def pool_pairs(records: Sequence[dict[str, Any]]) -> dict[str, set[str]]:
    """The pooled researchers per query: every system's ranked researchers."""
    pool: dict[str, set[str]] = {}
    for record in records:
        pool.setdefault(record["query_id"], set()).update(entry["author_id"] for entry in record["ranking"])
    return pool


def fair_evaluate(
    records: Sequence[dict[str, Any]],
    judgments: Judgments,
    settings: FairEvaluationSettings = FairEvaluationSettings(),
    pool_records: Sequence[dict[str, Any]] | None = None,
) -> FairEvaluationResult:
    """Evaluate every system on the queries whose pools are judged enough.

    ``pool_records`` (default: ``records``) defines the judged pool and so
    which queries are eligible. Passing the frozen Phase 1 records lets a
    later system be scored on the same queries; any researcher it returns that
    was never judged is removed from its ranking and counted in
    ``unjudged_removed``, never treated as irrelevant.
    """
    pool = pool_pairs(pool_records if pool_records is not None else records)
    coverage = {
        query_id: (sum((query_id, author) in judgments for author in authors), len(authors))
        for query_id, authors in sorted(pool.items())
    }
    result = FairEvaluationResult(
        total_pairs=sum(total for _, total in coverage.values()),
        judged_pairs=sum(judged for judged, _ in coverage.values()),
        coverage_by_query=coverage,
    )
    for query_id, (judged, total) in coverage.items():
        relevant = {
            author for author in pool[query_id]
            if judgments.get((query_id, author), -1) >= settings.relevant_threshold
        }
        if judged / total < settings.min_judged_fraction:
            result.excluded_queries[query_id] = f"only {judged} of {total} pooled pairs judged"
        elif not relevant:
            result.excluded_queries[query_id] = "no pooled researcher judged relevant"
        else:
            result.evaluated_queries.append(query_id)
    evaluated = set(result.evaluated_queries)
    for record in records:
        if record["query_id"] in evaluated:
            result.per_query.append(_score(record, judgments, settings))
    result.summary = _summarize(result.per_query)
    return result


def _score(record: dict[str, Any], judgments: Judgments, settings: FairEvaluationSettings) -> dict[str, Any]:
    query_id = record["query_id"]
    ranked = [entry["author_id"] for entry in record["ranking"]]
    condensed = [author for author in ranked if (query_id, author) in judgments]
    pool_levels = {
        author: level for (qid, author), level in judgments.items() if qid == query_id
    }
    relevant = frozenset(author for author, level in pool_levels.items() if level >= settings.relevant_threshold)
    metrics = {name: metric(condensed, relevant) for name, metric in METRICS.items()}
    metrics["ndcg@10_graded"] = _graded_ndcg(condensed, pool_levels)
    top = condensed[:_DEPTH]
    return {
        "system": record["system"],
        "query_id": query_id,
        "query": record["query"],
        "unjudged_removed": len(ranked) - len(condensed),
        **metrics,
        "judged_results": len(top),
        "relevant_results": sum(author in relevant for author in top),
    }


def _graded_ndcg(ranked: Sequence[str], levels: dict[str, int]) -> float:
    """nDCG@10 with the judged 0–3 level as each researcher's gain."""
    gain = sum(levels[author] / math.log2(rank + 1) for rank, author in enumerate(ranked[:_DEPTH], start=1))
    ideal_levels = sorted(levels.values(), reverse=True)[:_DEPTH]
    ideal = sum(level / math.log2(rank + 1) for rank, level in enumerate(ideal_levels, start=1))
    return gain / ideal if ideal else 0.0


def _summarize(per_query: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    systems: dict[str, list[dict[str, Any]]] = {}
    for row in per_query:
        systems.setdefault(row["system"], []).append(row)
    metric_names = [*METRICS, "ndcg@10_graded"]
    summary = []
    for system, rows in systems.items():
        judged = sum(row["judged_results"] for row in rows)
        relevant = sum(row["relevant_results"] for row in rows)
        summary.append(
            {
                "system": system,
                "queries": len(rows),
                **{name: round(statistics.fmean(row[name] for row in rows), 4) for name in metric_names},
                "unjudged_removed": sum(row["unjudged_removed"] for row in rows),
                "judged_results": judged,
                # Share of the judged top-10 results that are relevant; unaffected by removed results.
                "precision_among_judged": round(relevant / judged, 4) if judged else None,
            }
        )
    return summary
