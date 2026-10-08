"""Builds the manual judging file from a frozen Phase 1 run (read-only).

One row per pooled query-researcher pair, exactly the pairs in the run's
``candidate_pool.csv``. Every Phase 1 column is kept (renamed per
``PHASE1_COLUMN_MAP``), and judging context is added:

* scores from every system that returned the researcher;
* Week 4 evidence papers (the query-specific papers that matched);
* the researcher's Week 2 papers, so researchers returned only by Week 3,
  which exposes no evidence, can still be judged.

``existing_label`` is ``1`` for the 80 TF-IDF-derived benchmark labels and
blank otherwise (blank means "not in the old benchmark", never "irrelevant").
``manual_relevance`` starts blank on every row.
"""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.week4.config import CONFIG
from src.week4.data_adapter import Week2PaperSource
from src.week5.judging.schema import COLUMNS, EDITABLE_COLUMNS, PHASE1_COLUMN_MAP

_MAX_PROFILE_TITLES = 5
_TITLE_SEPARATOR = " || "


def build_template(run_dir: Path) -> pd.DataFrame:
    """Return the judging rows for every pooled pair of a frozen run, in pool order.

    Raises:
        FileNotFoundError: If the run lacks ``candidate_pool.csv`` or ``rankings_evidence.json``.
    """
    pool_path, evidence_path = run_dir / "candidate_pool.csv", run_dir / "rankings_evidence.json"
    for path in (pool_path, evidence_path):
        if not path.is_file():
            raise FileNotFoundError(f"Frozen Phase 1 artifact not found: {path}")
    pool = pd.read_csv(pool_path, dtype=str, keep_default_na=False)
    records: list[dict[str, Any]] = json.loads(evidence_path.read_text(encoding="utf-8"))
    scores, evidence = _scores_and_evidence(records)
    profiles = _profile_papers(set(pool["author_id"]))

    frame = pool.rename(columns=PHASE1_COLUMN_MAP)
    frame["existing_label"] = frame["existing_label"].map({"True": "1", "False": ""})
    keys = list(zip(frame["query_id"], frame["researcher_id"], strict=True))
    frame["score"] = [_join_scores(scores.get(key, {})) for key in keys]
    frame["evidence_paper_ids"] = [" | ".join(paper["paper_id"] for paper in evidence.get(key, [])) for key in keys]
    frame["evidence_paper_titles"] = [
        _TITLE_SEPARATOR.join(paper["title"] or "(untitled)" for paper in evidence.get(key, [])) for key in keys
    ]
    frame["profile_paper_count"] = [str(len(profiles.get(author, []))) for author in frame["researcher_id"]]
    frame["profile_paper_titles"] = [
        _TITLE_SEPARATOR.join(profiles.get(author, [])[:_MAX_PROFILE_TITLES]) for author in frame["researcher_id"]
    ]
    for column in EDITABLE_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[list(COLUMNS)]


def _scores_and_evidence(
    records: Sequence[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, float]], dict[tuple[str, str], list[dict[str, Any]]]]:
    """Collect each pair's score per system and its Week 4 evidence papers (best first, unique)."""
    scores: dict[tuple[str, str], dict[str, float]] = {}
    evidence: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        for entry in record["ranking"]:
            key = (record["query_id"], entry["author_id"])
            if entry["score"] is not None:
                scores.setdefault(key, {})[record["system"]] = entry["score"]
            papers = evidence.setdefault(key, {})
            for paper in entry["evidence"]:
                papers.setdefault(paper["paper_id"], paper)
    ordered = {
        key: sorted(papers.values(), key=lambda paper: (-paper["score"], paper["paper_id"]))
        for key, papers in evidence.items()
    }
    return scores, ordered


def _join_scores(system_scores: dict[str, float]) -> str:
    return "; ".join(f"{system}={score:.4f}" for system, score in system_scores.items())


def _profile_papers(author_ids: Iterable[str]) -> dict[str, list[str]]:
    """Each researcher's Week 2 papers as ``"year: title"``, newest first (read via Week 4's adapter)."""
    wanted = set(author_ids)
    papers: dict[str, list[tuple[int, str]]] = {}
    source = Week2PaperSource(CONFIG.papers_path, CONFIG.authorship_path)
    for paper in source.load_papers():
        year = paper.metadata.get("publication_year")
        year_value = year if isinstance(year, int) else 0
        for author in paper.authors:
            if author.author_id in wanted:
                papers.setdefault(author.author_id, []).append((year_value, paper.title or "(untitled)"))
    return {
        author: [f"{year or 'n.d.'}: {title}" for year, title in sorted(items, key=lambda item: (-item[0], item[1]))]
        for author, items in papers.items()
    }
