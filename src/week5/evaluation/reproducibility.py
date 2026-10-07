"""Compare two frozen runs to check whether the evaluation is reproducible.

Determinism is only reported when it is actually observed: identical bytes,
identical aggregate metrics and identical rankings, on the same index.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.week5.evaluation.frozen_run import (
    DETERMINISTIC_FILES,
    EVIDENCE_JSON,
    METRICS_JSON,
    RUN_METADATA_FILE,
)
from src.week5.paths import relative_to_root


def compare_runs(run_a: Path, run_b: Path) -> dict[str, Any]:
    """Compare two run folders: index used, file bytes, aggregate metrics, rankings and scores.

    Raises:
        FileNotFoundError: If either folder lacks the files a run writes.
    """
    for run in (run_a, run_b):
        missing = [name for name in (*DETERMINISTIC_FILES, RUN_METADATA_FILE) if not (run / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{run} is not a complete run; missing {', '.join(missing)}.")
    meta_a, meta_b = (_load(run / RUN_METADATA_FILE) for run in (run_a, run_b))
    rankings_a, rankings_b = (_rankings(_load(run / EVIDENCE_JSON)) for run in (run_a, run_b))
    differing = sorted(key for key in rankings_a.keys() | rankings_b.keys() if rankings_a.get(key) != rankings_b.get(key))
    report = {
        "compared_utc": datetime.now(UTC).isoformat(),
        "run_a": relative_to_root(run_a),
        "run_b": relative_to_root(run_b),
        "same_index": meta_a["index_before"] == meta_b["index_before"],
        "index_unchanged_within_each_run": meta_a["index_unchanged_during_run"] and meta_b["index_unchanged_during_run"],
        "aggregate_metrics_identical": _load(run_a / METRICS_JSON) == _load(run_b / METRICS_JSON),
        "rankings_identical": not differing,
        "differing_rankings": [{"system": system, "query_id": query_id} for system, query_id in differing],
        "max_score_difference": _max_score_difference(rankings_a, rankings_b),
        "files_byte_identical": {name: _sha256(run_a / name) == _sha256(run_b / name) for name in DETERMINISTIC_FILES},
    }
    report["deterministic"] = (
        report["same_index"]
        and report["aggregate_metrics_identical"]
        and report["rankings_identical"]
        and all(report["files_byte_identical"].values())
    )
    return report


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Write the comparison as JSON and as a short Markdown summary; never overwrite."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(report['run_a']).name}_vs_{Path(report['run_b']).name}"
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    for path in (json_path, md_path):
        if path.exists():
            raise FileExistsError(f"{path} already exists; delete it to compare these runs again.")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    return json_path, md_path


def _rankings(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[tuple[str, float]]]:
    return {
        (record["system"], record["query_id"]): [(entry["author_id"], entry["score"]) for entry in record["ranking"]]
        for record in records
    }


def _max_score_difference(
    a: dict[tuple[str, str], list[tuple[str, float]]], b: dict[tuple[str, str], list[tuple[str, float]]]
) -> float | None:
    """Largest score difference for the same researcher at the same rank, or None if none match."""
    differences = [
        abs(score_a - score_b)
        for key in a.keys() & b.keys()
        for (author_a, score_a), (author_b, score_b) in zip(a[key], b[key])
        if author_a == author_b
    ]
    return max(differences) if differences else None


def _markdown(report: dict[str, Any]) -> str:
    verdict = "deterministic" if report["deterministic"] else "NOT identical"
    lines = [
        f"# Reproducibility check: {Path(report['run_a']).name} vs {Path(report['run_b']).name}",
        "",
        f"**Verdict: {verdict}.**",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Same Week 4 index (count, file sizes, vector segments) | {report['same_index']} |",
        f"| Index unchanged during each run | {report['index_unchanged_within_each_run']} |",
        f"| Aggregate metrics identical | {report['aggregate_metrics_identical']} |",
        f"| Rankings identical (all systems, all queries) | {report['rankings_identical']} |",
        f"| Largest score difference | {report['max_score_difference']} |",
    ]
    lines += [f"| `{name}` byte-identical | {same} |" for name, same in report["files_byte_identical"].items()]
    if report["differing_rankings"]:
        lines += ["", "Differing rankings:", ""]
        lines += [f"- {item['system']}, {item['query_id']}" for item in report["differing_rankings"]]
    return "\n".join(lines) + "\n"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
