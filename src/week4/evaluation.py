"""Compares researcher recommenders on labelled queries.

Evaluates any object implementing ``ResearcherRecommender``: the Week 3
baseline (via ``baseline_adapter``) and one Week 4 pipeline per ranking
strategy (via ``build_week4_pipeline``), all in one run. It compares systems,
not implementations: it never touches ChromaDB, Week 2 CSV schemas or Week 3
code.

Metrics operate only on ranked researcher IDs. Raw similarity scores are
never compared, because Week 3 and Week 4 scores are on different scales.

Evaluation is deterministic: no metric uses randomness, systems and queries
are processed in the order given, and repeated runs over identical inputs
produce identical outputs.

Outputs, written to ``config.results_dir``:

* Per-query results: one row per (system, query) with the query text, the
  system's ranked researcher IDs and every metric value.
* Summary: the mean of every metric per system. Because each Week 4 ranking
  strategy runs as its own named pipeline, this is also the strategy comparison.

Run from the repository root::

    python -m src.week4.evaluation
"""

# TODO(Phase 2): pooling bias. Against the labels, each returned researcher is
# judged relevant, judged non-relevant or unjudged (never labelled). Week 2's
# benchmark (create_evaluation_set.py) labels only the TF-IDF top 10 per
# query, and labels all of them relevant, so researchers found only by other
# systems are unjudged and the benchmark favours TF-IDF.
# Unjudged researchers must never be silently treated as non-relevant: either
# state that assumption alongside every reported metric, or also report how
# many of each system's top results are judged. LabeledQuery may then need
# the judged non-relevant IDs as well.

import dataclasses
import logging
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.week4 import data_adapter
from src.week4.baseline_adapter import BaselineMode, Week3Baseline
from src.week4.config import CONFIG, RankingStrategyName, Week4Config
from src.week4.interfaces import ResearcherRecommender
from src.week4.schemas import LabeledQuery
from src.week4.week4_pipeline import build_week4_pipeline

logger = logging.getLogger(__name__)

# Deepest cutoff used by any metric below; every system is asked for exactly
# this many researchers so all systems are evaluated at the same depth.
EVALUATION_DEPTH = 10

# Week 3 modes, primary comparison (semantic) first.
BASELINE_MODES: tuple[BaselineMode, ...] = ("semantic", "tfidf", "hybrid")
# "weighted" has no formula yet, so it is not evaluated.
WEEK4_STRATEGIES: tuple[RankingStrategyName, ...] = ("best_paper", "average", "top_k_average")

PER_QUERY_FILENAME = "per_query_results.csv"
SUMMARY_FILENAME = "summary.csv"
# TODO(future): reserved for per_strategy_summary.csv and metric_breakdown.csv.


@dataclass(frozen=True, slots=True)
class QueryResult:
    """One system's result for one query.

    Attributes:
        system: ``ResearcherRecommender.name`` of the evaluated system.
        query: Query text.
        ranked_author_ids: Returned researcher IDs, best-first.
        metrics: Metric name -> value for this query.
    """

    system: str
    query: str
    ranked_author_ids: tuple[str, ...]
    metrics: Mapping[str, float] = field(default_factory=dict, hash=False)


# -------------------------
# Metrics
# -------------------------
# Each metric is a pure, deterministic function of the ranked researcher IDs
# (best-first) and the set of relevant IDs for one query. Relevance is binary
# until graded labels exist. Documented assumption: every returned researcher
# not in the relevant set counts as non-relevant, including unjudged ones
# (see the pooling-bias TODO above).


def _relevant_in_top(ranked_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> int:
    """Count the relevant researchers among the top ``k``."""
    return sum(1 for author_id in ranked_ids[:k] if author_id in relevant_ids)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> float:
    """Fraction of the top ``k`` positions holding a relevant researcher.

    Always divides by ``k``, so a system returning fewer than ``k`` results
    is penalized rather than rewarded.
    """
    return _relevant_in_top(ranked_ids, relevant_ids, k) / k


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> float:
    """Fraction of all relevant researchers found in the top ``k``.

    ``relevant_ids`` must not be empty; ``evaluate_all`` excludes such queries.
    """
    return _relevant_in_top(ranked_ids, relevant_ids, k) / len(relevant_ids)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: frozenset[str]) -> float:
    """1 / rank of the first relevant researcher, or 0 if none is returned.

    Averaged over queries in the summary, this is the MRR.
    """
    for rank, author_id in enumerate(ranked_ids, start=1):
        if author_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> float:
    """Normalized discounted cumulative gain over the top ``k`` (binary gains).

    ``relevant_ids`` must not be empty; ``evaluate_all`` excludes such queries.
    """
    gain = sum(
        1.0 / math.log2(rank + 1)
        for rank, author_id in enumerate(ranked_ids[:k], start=1)
        if author_id in relevant_ids
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant_ids)) + 1))
    return gain / ideal


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> float:
    """1.0 if any relevant researcher appears in the top ``k``, else 0.0."""
    return 1.0 if _relevant_in_top(ranked_ids, relevant_ids, k) else 0.0


MetricFunction = Callable[[Sequence[str], frozenset[str]], float]

# Every reported metric, in output column order. To add a metric, write its
# function above and register it here.
METRICS: dict[str, MetricFunction] = {
    "precision@5": lambda ranked, relevant: precision_at_k(ranked, relevant, 5),
    "precision@10": lambda ranked, relevant: precision_at_k(ranked, relevant, 10),
    "recall@10": lambda ranked, relevant: recall_at_k(ranked, relevant, 10),
    "mrr": reciprocal_rank,
    "ndcg@10": lambda ranked, relevant: ndcg_at_k(ranked, relevant, 10),
    "hit@10": lambda ranked, relevant: hit_at_k(ranked, relevant, 10),
}
# TODO(future): possible additional metrics: MAP, Recall@20, Precision@20 and
# Success@1. Any cutoff deeper than EVALUATION_DEPTH requires raising it.


# -------------------------
# Runner
# -------------------------


def evaluate_recommender(
    recommender: ResearcherRecommender, queries: Sequence[LabeledQuery]
) -> list[QueryResult]:
    """Run one recommender over every query and compute every metric.

    Each query is sent with ``top_k=EVALUATION_DEPTH``; only the returned
    researcher IDs are kept, never their scores. Results follow the order
    of ``queries``.

    If the recommender fails on a query, the system and query are logged and
    the original exception is re-raised unchanged, aborting the run.
    """
    results = []
    for labeled_query in queries:
        try:
            ranking = recommender.recommend(labeled_query.query, top_k=EVALUATION_DEPTH)
        except Exception:
            logger.error("%s failed on query %r.", recommender.name, labeled_query.query)
            raise
        ranked_ids = tuple(result.author.author_id for result in ranking)
        results.append(
            QueryResult(
                system=recommender.name,
                query=labeled_query.query,
                ranked_author_ids=ranked_ids,
                metrics={
                    name: metric(ranked_ids, labeled_query.relevant_author_ids)
                    for name, metric in METRICS.items()
                },
            )
        )
    return results


def evaluate_all(
    recommenders: Sequence[ResearcherRecommender], queries: Sequence[LabeledQuery]
) -> list[QueryResult]:
    """Evaluate several recommenders on the same queries.

    Results preserve the order of ``recommenders`` and, within each system,
    the order of ``queries``, so result tables are deterministic and
    reproducible.

    Queries with no relevant researchers are excluded (recall is undefined
    for them) and their count is logged as a warning.

    Raises:
        ValueError: If ``recommenders`` or the usable queries are empty, or
            two recommenders share a ``name``.
    """
    if not recommenders:
        raise ValueError("There are no recommenders to evaluate.")
    names = [recommender.name for recommender in recommenders]
    if len(set(names)) != len(names):
        raise ValueError(f"Recommender names must be unique, got {names}.")
    usable = [query for query in queries if query.relevant_author_ids]
    if len(usable) < len(queries):
        logger.warning(
            "Excluded %d of %d queries that have no relevant researcher.",
            len(queries) - len(usable),
            len(queries),
        )
    if not usable:
        raise ValueError("No labelled query has a relevant researcher.")
    return [
        result for recommender in recommenders for result in evaluate_recommender(recommender, usable)
    ]


def summarize(results: Sequence[QueryResult]) -> list[dict[str, str | float | int]]:
    """Compute one summary row per system: name, query count and mean of every metric.

    Rows follow the order in which systems were evaluated.
    """
    by_system: dict[str, list[QueryResult]] = {}
    for result in results:
        by_system.setdefault(result.system, []).append(result)
    summary: list[dict[str, str | float | int]] = []
    for system, system_results in by_system.items():
        row: dict[str, str | float | int] = {"system": system, "queries": len(system_results)}
        for name in METRICS:
            row[name] = round(statistics.fmean(result.metrics[name] for result in system_results), 4)
        summary.append(row)
    return summary


def write_results(
    results: Sequence[QueryResult],
    summary: Sequence[Mapping[str, str | float | int]],
    results_dir: Path,
) -> None:
    """Write per-query results and the summary as CSV files in ``results_dir``.

    Creates ``results_dir`` if it does not exist. Ranked IDs are written in
    order so baseline and Week 4 rankings can be compared query by query.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    per_query = [
        {
            "system": result.system,
            "query": result.query,
            **{name: round(value, 4) for name, value in result.metrics.items()},
            "ranked_author_ids": "|".join(result.ranked_author_ids),
        }
        for result in results
    ]
    pd.DataFrame(per_query).to_csv(results_dir / PER_QUERY_FILENAME, index=False)
    pd.DataFrame(list(summary)).to_csv(results_dir / SUMMARY_FILENAME, index=False)


def build_recommenders(
    config: Week4Config,
    *,
    baseline_modes: Sequence[BaselineMode],
    strategies: Sequence[RankingStrategyName],
) -> list[ResearcherRecommender]:
    """Build every system to compare: one baseline per mode, one Week 4 pipeline per strategy.

    Week 4 pipelines are obtained only through ``build_week4_pipeline``, using
    a copy of ``config`` with ``ranking_strategy`` replaced.
    """
    # Construction order: baseline adapters first, then Week 4 pipelines. The
    # returned list keeps that order when passed to evaluate_all(), so
    # baselines come first in every result table.
    recommenders: list[ResearcherRecommender] = [
        Week3Baseline(mode=mode) for mode in baseline_modes
    ]
    recommenders.extend(
        build_week4_pipeline(dataclasses.replace(config, ranking_strategy=strategy))
        for strategy in strategies
    )
    return recommenders


def load_labeled_queries(config: Week4Config) -> list[LabeledQuery]:
    """Load the labelled queries from ``config.evaluation_labels_path``.

    The label file is Week 2 output, so reading it is delegated to
    ``data_adapter``.

    Raises:
        ValueError: If ``config.evaluation_labels_path`` is not set or the
            file is malformed.
        FileNotFoundError: If the label file does not exist.
    """
    return data_adapter.load_labeled_queries(config.evaluation_labels_path)


def main(config: Week4Config = CONFIG) -> None:
    """Evaluate the baseline and Week 4 strategies and write the results."""
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Third-party libraries stay at WARNING; Week 4 modules, including this one
    # when it runs as __main__, log at the configured level.
    for name in ("src.week4", __name__):
        logging.getLogger(name).setLevel(config.log_level)

    start = time.perf_counter()
    queries = load_labeled_queries(config)
    recommenders = build_recommenders(
        config, baseline_modes=BASELINE_MODES, strategies=WEEK4_STRATEGIES
    )
    logger.info("Evaluating %d systems on %d labelled queries.", len(recommenders), len(queries))
    logger.warning(
        "Researchers absent from the labels count as non-relevant, which favours the system "
        "that produced the labelled candidates (see the pooling-bias note in src/week4/README.md)."
    )
    results = evaluate_all(recommenders, queries)
    summary = summarize(results)
    write_results(results, summary, config.results_dir)
    print(pd.DataFrame(summary).to_string(index=False))
    logger.info(
        "Evaluation finished in %.1fs; results written to %s.",
        time.perf_counter() - start,
        config.results_dir,
    )


if __name__ == "__main__":
    main()
