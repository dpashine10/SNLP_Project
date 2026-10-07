"""Frozen Week 3 vs Week 4 evaluation that keeps full evidence for every ranking.

The systems, metrics and aggregation all come from ``src.week4.evaluation``
(``build_recommenders``, ``METRICS``, ``summarize``), so the algorithms are
exactly those of Week 4's own evaluation. The difference is what is kept:
Week 4's runner stores only ranked IDs, while this one also stores names,
scores and Week 4's evidence papers. Each run is written to a new folder
under ``src/week5/artifacts/runs/`` and never overwrites an earlier run.
"""

import json
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.week4.baseline_adapter import Week3Baseline
from src.week4.config import Week4Config
from src.week4.evaluation import (
    BASELINE_MODES,
    EVALUATION_DEPTH,
    METRICS,
    SUMMARY_FILENAME,
    WEEK4_STRATEGIES,
    QueryResult,
    build_recommenders,
    load_labeled_queries,
    summarize,
)
from src.week4.interfaces import ResearcherRecommender
from src.week4.schemas import LabeledQuery
from src.week5.evaluation.environment import (
    capture_environment,
    check_inputs,
    index_fingerprint,
    load_week3_module,
)
from src.week5.evaluation.evidence import candidate_pool, ranking_record
from src.week5.paths import RUNS_DIR, relative_to_root

logger = logging.getLogger(__name__)

RUN_COMMAND = "python -m src.week5.scripts.freeze_evaluation"
RUN_METADATA_FILE = "run_metadata.json"
METRICS_CSV = "metrics_summary.csv"
METRICS_JSON = "metrics_summary.json"
PER_QUERY_CSV = "per_query_metrics.csv"
EVIDENCE_JSON = "rankings_evidence.json"
CANDIDATE_POOL_CSV = "candidate_pool.csv"
LOG_FILE = "run.log"
# Outputs that depend only on the systems and the data, not on the run time.
DETERMINISTIC_FILES = (METRICS_CSV, METRICS_JSON, PER_QUERY_CSV, EVIDENCE_JSON, CANDIDATE_POOL_CSV)


def run_frozen_evaluation(config: Week4Config, runs_dir: Path = RUNS_DIR) -> Path:
    """Evaluate all six systems on the existing benchmark and write a new run folder.

    Returns:
        The run folder.

    Raises:
        FileNotFoundError: If a Week 2 or Week 3 input is missing.
        RuntimeError: If the Week 4 index is missing or empty.
    """
    week3 = load_week3_module()
    check_inputs(config, week3)
    run_dir = _new_run_dir(runs_dir)
    handler = _attach_log_file(run_dir / LOG_FILE)
    started = datetime.now(UTC)
    try:
        logger.info("Run %s started; writing to %s.", run_dir.name, relative_to_root(run_dir))
        index_before = index_fingerprint(config)
        queries = load_labeled_queries(config)
        environment = capture_environment(config, week3, queries)
        records, query_results = _evaluate(config, queries)
        summary = summarize(query_results)
        index_after = index_fingerprint(config)
        _write_outputs(run_dir, records, query_results, summary)
        metadata = {
            "run_id": run_dir.name,
            "command": RUN_COMMAND,
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(UTC).isoformat(),
            "systems": list(dict.fromkeys(result.system for result in query_results)),
            "index_before": index_before,
            "index_after": index_after,
            "index_unchanged_during_run": index_before == index_after,
            "matches_week4_results": _compare_with_week4_results(summary, config),
            "environment": environment,
            "files": [*DETERMINISTIC_FILES, LOG_FILE],
        }
        _write_json(run_dir / RUN_METADATA_FILE, metadata)
        logger.info("Run %s finished.", run_dir.name)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return run_dir


def _evaluate(
    config: Week4Config, queries: Sequence[LabeledQuery]
) -> tuple[list[dict[str, Any]], list[QueryResult]]:
    """Run every system on every usable query, keeping full rankings and Week 4 metrics."""
    usable = [query for query in queries if query.relevant_author_ids]
    if len(usable) < len(queries):
        logger.warning("Excluded %d queries with no relevant researcher.", len(queries) - len(usable))
    recommenders = build_recommenders(
        config, baseline_modes=BASELINE_MODES, strategies=WEEK4_STRATEGIES
    )
    records: list[dict[str, Any]] = []
    query_results: list[QueryResult] = []
    for recommender in recommenders:
        start = time.perf_counter()
        for index, labeled_query in enumerate(usable, start=1):
            ranking = recommender.recommend(labeled_query.query, top_k=EVALUATION_DEPTH)
            ranked_ids = tuple(result.author.author_id for result in ranking)
            metrics = {
                name: metric(ranked_ids, labeled_query.relevant_author_ids)
                for name, metric in METRICS.items()
            }
            query_results.append(
                QueryResult(
                    system=recommender.name,
                    query=labeled_query.query,
                    ranked_author_ids=ranked_ids,
                    metrics=metrics,
                )
            )
            records.append(
                ranking_record(
                    f"q{index:02d}", labeled_query, recommender.name, _family(recommender), ranking, metrics
                )
            )
        logger.info("%s evaluated in %.1fs.", recommender.name, time.perf_counter() - start)
    return records, query_results


def _family(recommender: ResearcherRecommender) -> str:
    return "week3" if isinstance(recommender, Week3Baseline) else "week4"


def _write_outputs(
    run_dir: Path,
    records: list[dict[str, Any]],
    query_results: list[QueryResult],
    summary: list[dict[str, str | float | int]],
) -> None:
    pd.DataFrame(summary).to_csv(run_dir / METRICS_CSV, index=False)
    _write_json(run_dir / METRICS_JSON, summary)
    query_ids = {record["query"]: record["query_id"] for record in records}
    per_query = [
        {
            "system": result.system,
            "query_id": query_ids[result.query],
            "query": result.query,
            **result.metrics,
            "ranked_author_ids": "|".join(result.ranked_author_ids),
        }
        for result in query_results
    ]
    pd.DataFrame(per_query).to_csv(run_dir / PER_QUERY_CSV, index=False)
    _write_json(run_dir / EVIDENCE_JSON, records)
    pd.DataFrame(candidate_pool(records)).to_csv(run_dir / CANDIDATE_POOL_CSV, index=False)


def _compare_with_week4_results(
    summary: list[dict[str, str | float | int]], config: Week4Config
) -> dict[str, Any]:
    """Check whether this run reproduces Week 4's saved summary (read-only)."""
    path = config.results_dir / SUMMARY_FILENAME
    if not path.is_file():
        return {"available": False, "path": relative_to_root(path)}
    saved = pd.read_csv(path)
    current = pd.DataFrame(summary)
    identical = list(saved.columns) == list(current.columns) and saved.equals(
        current.astype(saved.dtypes.to_dict())
    )
    return {"available": True, "path": relative_to_root(path), "identical": identical}


def _new_run_dir(runs_dir: Path) -> Path:
    """Create a fresh run folder named by UTC time; never reuse an existing one."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for suffix in ("", *(f"-{n}" for n in range(2, 100))):
        candidate = runs_dir / f"{stamp}{suffix}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"Could not create a new run folder under {runs_dir}.")


def _attach_log_file(path: Path) -> logging.Handler:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    return handler


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
