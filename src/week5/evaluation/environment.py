"""Captures the configuration and environment behind a frozen evaluation run.

Everything here is read-only: it reads Week 2 data, the Week 3 module's
settings and the existing Week 4 index, and never changes any of them.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from src.week4.config import PROJECT_ROOT, Week4Config
from src.week4.data_adapter import Week2PaperSource
from src.week4.evaluation import BASELINE_MODES, EVALUATION_DEPTH, METRICS, WEEK4_STRATEGIES
from src.week4.schemas import LabeledQuery
from src.week4.vector_store import VectorStore
from src.week5.paths import relative_to_root

WEEK3_MODULE = "src.week2.pipeline"
_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "sentence-transformers",
    "transformers",
    "torch",
    "chromadb",
)


def load_week3_module() -> ModuleType:
    """Import the Week 3 baseline module (``ResearchDiscoveryPipeline`` lives there)."""
    return importlib.import_module(WEEK3_MODULE)


def check_inputs(config: Week4Config, week3: ModuleType) -> None:
    """Fail early with a clear message if any input of the frozen evaluation is missing.

    Raises:
        FileNotFoundError: If a Week 2 file or a Week 3 embedding file is missing.
        RuntimeError: If the Week 4 ChromaDB index is missing or empty.
    """
    required = {
        "Week 2 papers": config.papers_path,
        "Week 2 paper-author links": config.authorship_path,
        "Week 2 benchmark labels": config.evaluation_labels_path,
        "Week 3 researcher profiles": week3.PROFILES_PATH,
        "Week 3 paper embeddings": week3.PAPER_EMB_PATH,
        "Week 3 researcher embeddings": week3.RESEARCHER_EMB_PATH,
    }
    for label, path in required.items():
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if index_fingerprint(config)["papers"] == 0:
        raise RuntimeError(
            f"The Week 4 index at {config.chroma_dir} is missing or empty. "
            "Run `python -m src.week4.build_chromadb` first."
        )


def index_fingerprint(config: Week4Config) -> dict[str, Any]:
    """Identify the Week 4 index without copying it: location, paper count and file sizes.

    The sqlite file's modification time is left out on purpose: ChromaDB
    updates it whenever the index is opened, even for read-only use. The
    vector-segment files are only written by indexing, so their size and
    latest modification time identify a particular build.
    """
    store = VectorStore(
        config.chroma_dir,
        config.collection_name,
        distance_metric=config.distance_metric,
        embedding_model_name=config.embedding_model_name,
    )
    sqlite_file = config.chroma_dir / "chroma.sqlite3"
    segment_files = [
        path for path in config.chroma_dir.glob("*/*") if path.is_file()
    ]
    latest_segment_write = max((path.stat().st_mtime for path in segment_files), default=None)
    return {
        "path": relative_to_root(config.chroma_dir),
        "collection": config.collection_name,
        "papers": store.count(),
        "sqlite_bytes": sqlite_file.stat().st_size if sqlite_file.is_file() else None,
        "vector_segment_bytes": sum(path.stat().st_size for path in segment_files),
        "vector_segment_modified_utc": (
            datetime.fromtimestamp(latest_segment_write, UTC).isoformat()
            if latest_segment_write is not None
            else None
        ),
    }


def capture_environment(
    config: Week4Config, week3: ModuleType, queries: list[LabeledQuery]
) -> dict[str, Any]:
    """Record everything needed to reproduce and interpret a frozen run."""
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "git": _git_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in _PACKAGES},
        "device": _device_state(config),
        "dataset": _dataset_counts(config, week3),
        "index": index_fingerprint(config),
        "retrieval": {
            "embedding_model": config.embedding_model_name,
            "distance_metric": config.distance_metric,
            "papers_retrieved_per_query": config.paper_top_k,
            "researchers_evaluated_per_query": EVALUATION_DEPTH,
            "top_k_average_k": config.strategy_top_k,
        },
        "systems": {
            "week3_module": f"{WEEK3_MODULE}.ResearchDiscoveryPipeline",
            "week3_embedding_model": week3.MODEL_NAME,
            "week3_modes": list(BASELINE_MODES),
            "week3_hybrid_alpha": inspect.signature(
                week3.ResearchDiscoveryPipeline.search
            ).parameters["alpha"].default,
            "week3_embeddings": [
                relative_to_root(week3.PAPER_EMB_PATH),
                relative_to_root(week3.RESEARCHER_EMB_PATH),
            ],
            "week4_strategies": list(WEEK4_STRATEGIES),
        },
        "benchmark": _benchmark_summary(config, queries),
        "metrics": list(METRICS),
        "seeds": {
            "week4_config_random_seed": config.random_seed,
            "note": "No retrieval, ranking or evaluation step uses randomness; the seed is recorded for completeness.",
        },
    }


def _dataset_counts(config: Week4Config, week3: ModuleType) -> dict[str, int]:
    """Count papers and researchers, and the papers Week 4's adapter accepts for indexing."""
    assert config.papers_path is not None
    papers = len(pd.read_csv(config.papers_path, usecols=["work_id"], dtype=str))
    researchers = len(pd.read_csv(week3.PROFILES_PATH, usecols=["author_id"], dtype=str))
    source = Week2PaperSource(config.papers_path, config.authorship_path)
    indexable = sum(1 for _ in source.load_papers())
    return {
        "papers": papers,
        "researchers": researchers,
        "papers_indexable": indexable,
        "papers_skipped": papers - indexable,
    }


def _benchmark_summary(config: Week4Config, queries: list[LabeledQuery]) -> dict[str, Any]:
    """Describe the benchmark labels exactly as they are stored."""
    assert config.evaluation_labels_path is not None
    labels = pd.read_csv(
        config.evaluation_labels_path, usecols=["query", "relevant"], dtype=str
    )
    return {
        "labels_path": relative_to_root(config.evaluation_labels_path),
        "origin": "src/week2/create_evaluation_set.py (TF-IDF top 10 per query)",
        "queries": len(queries),
        "labelled_rows": len(labels),
        "rows_labelled_relevant": int((labels["relevant"].str.strip() == "1").sum()),
        "relevant_per_query": {query.query: len(query.relevant_author_ids) for query in queries},
    }


def _device_state(config: Week4Config) -> dict[str, Any]:
    import torch

    return {
        "configured": config.device,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state() -> dict[str, Any]:
    try:
        commit = _git("rev-parse", "HEAD")
        status = _git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}
    return {"commit": commit, "uncommitted_changes": len(status.splitlines())}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
