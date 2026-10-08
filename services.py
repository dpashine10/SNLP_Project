"""Data preparation for the web interface.

Everything the interface shows is prepared here as plain Python data. This
module renders no HTML and imports no web framework, so it can be used and
tested without a server. Retrieval itself stays in
``src.week2.pipeline.ResearchDiscoveryPipeline`` and is called unchanged.
"""

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CALCULATED_DIR = PROJECT_ROOT / "Data" / "processed" / "calculated"

# Retrieval modes accepted by ResearchDiscoveryPipeline.search, with display labels.
MODE_LABELS: dict[str, str] = {
    "semantic": "Semantic (Sentence-BERT)",
    "tfidf": "TF-IDF (Keyword Baseline)",
    "hybrid": "Hybrid (Semantic + TF-IDF)",
}
DEFAULT_MODE = "semantic"
TOP_K_MIN, TOP_K_MAX, TOP_K_STEP, DEFAULT_TOP_K = 5, 25, 5, 10
ALPHA_MIN, ALPHA_MAX, ALPHA_STEP, DEFAULT_ALPHA = 0.0, 1.0, 0.05, 0.7

QUERY_PRESETS: tuple[tuple[str, str], ...] = (
    ("NLP & Information Extraction", "Application of Large Language Models to information extraction"),
    ("Transformer Classification", "transformer based text classification"),
    ("Knowledge Graph Reasoning", "knowledge graph embeddings and reasoning"),
    ("Cross-Lingual Retrieval", "cross-lingual information retrieval and neural search"),
)

COMPARISON_DEFAULT_QUERY = "transformer based text classification"
COMPARISON_TOP_K = 5
COMPARISON_MODES = ("semantic", "tfidf")


@dataclass(frozen=True)
class EvaluationTable:
    """A Week 2/3 metrics file shown on the evaluation page."""

    key: str
    title: str
    path: Path


EVALUATION_TABLES: tuple[EvaluationTable, ...] = (
    EvaluationTable("model_comparison", "Overall Model Performance Summary", CALCULATED_DIR / "model_comparison.csv"),
    EvaluationTable("baseline_metrics", "TF-IDF Baseline by Query", CALCULATED_DIR / "baseline_metrics.csv"),
    EvaluationTable("semantic_metrics", "Sentence Transformers by Query", CALCULATED_DIR / "semantic_metrics.csv"),
)


@dataclass(frozen=True)
class SearchSettings:
    """Validated search configuration."""

    mode: str = DEFAULT_MODE
    top_k: int = DEFAULT_TOP_K
    alpha: float = DEFAULT_ALPHA

    @property
    def mode_label(self) -> str:
        return MODE_LABELS[self.mode]

    @property
    def effective_alpha(self) -> float:
        """Alpha only applies to hybrid mode; other modes always pass the default."""
        return self.alpha if self.mode == "hybrid" else DEFAULT_ALPHA


@dataclass(frozen=True)
class Stat:
    """A dataset figure: counts are integers, formatted by the template."""

    label: str
    value: int | str


@dataclass(frozen=True)
class ResearcherResult:
    """One ranked researcher, ready for display."""

    rank: int
    name: str
    author_id: str
    author_url: str | None
    similarity_score: float
    n_papers: int
    paper_title: str
    paper_year: str
    paper_score: float
    topics: tuple[str, ...]


@dataclass(frozen=True)
class SearchOutcome:
    settings: SearchSettings
    results: tuple[ResearcherResult, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class ComparisonColumn:
    mode: str
    title: str
    results: tuple[ResearcherResult, ...]


@dataclass(frozen=True)
class TableData:
    """A CSV file as display strings, exactly as written in the file."""

    table: EvaluationTable
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


COMPARISON_TITLES: dict[str, str] = {
    "semantic": "Sentence Transformers (Dense Semantic)",
    "tfidf": "TF-IDF Baseline (Sparse Keyword)",
}


def parse_search_settings(args: Mapping[str, str]) -> tuple[SearchSettings, list[str]]:
    """Read settings from request parameters; invalid values fall back to defaults with an error."""
    errors: list[str] = []
    mode = args.get("mode", DEFAULT_MODE)
    if mode not in MODE_LABELS:
        errors.append(f"Unknown retrieval mode {mode!r}.")
        mode = DEFAULT_MODE

    top_k = DEFAULT_TOP_K
    raw_top_k = args.get("top_k")
    if raw_top_k is not None:
        try:
            top_k = int(raw_top_k)
            if not (TOP_K_MIN <= top_k <= TOP_K_MAX and top_k % TOP_K_STEP == 0):
                raise ValueError
        except ValueError:
            errors.append(f"Number of researchers must be one of {TOP_K_MIN}–{TOP_K_MAX} in steps of {TOP_K_STEP}.")
            top_k = DEFAULT_TOP_K

    alpha = DEFAULT_ALPHA
    raw_alpha = args.get("alpha")
    if raw_alpha is not None:
        try:
            alpha = float(raw_alpha)
            if not ALPHA_MIN <= alpha <= ALPHA_MAX:
                raise ValueError
        except ValueError:
            errors.append(f"Semantic weight must be between {ALPHA_MIN} and {ALPHA_MAX}.")
            alpha = DEFAULT_ALPHA
    return SearchSettings(mode=mode, top_k=top_k, alpha=alpha), errors


def to_researcher_result(raw: Mapping[str, Any]) -> ResearcherResult:
    """Convert one ``pipeline.search`` result into display data."""
    paper = raw["relevant_paper"]
    author_id = str(raw["author_id"])
    return ResearcherResult(
        rank=int(raw["rank"]),
        name=str(raw["author_name"]),
        author_id=author_id,
        author_url=author_id if author_id.startswith(("https://", "http://")) else None,
        similarity_score=float(raw["similarity_score"]),
        n_papers=int(raw["n_papers"]),
        paper_title=str(paper["title"]),
        paper_year=str(paper.get("year", "N/A")),
        paper_score=float(paper.get("paper_score", 0.0)),
        topics=tuple(str(topic) for topic in raw["related_topics"]),
    )


def run_search(pipeline: Any, query: str, settings: SearchSettings) -> SearchOutcome:
    """Rank researchers for ``query`` with the chosen settings, timing the call."""
    query = query.strip()
    start = time.time()
    raw = pipeline.search(query=query, top_k=settings.top_k, mode=settings.mode, alpha=settings.effective_alpha)
    elapsed = time.time() - start
    return SearchOutcome(settings, tuple(to_researcher_result(r) for r in raw), elapsed)


def run_comparison(pipeline: Any, query: str) -> list[ComparisonColumn]:
    """Top researchers for ``query`` from the semantic and TF-IDF modes, side by side."""
    return [
        ComparisonColumn(
            mode,
            COMPARISON_TITLES[mode],
            tuple(to_researcher_result(r) for r in pipeline.search(query=query, top_k=COMPARISON_TOP_K, mode=mode)),
        )
        for mode in COMPARISON_MODES
    ]


def dataset_stats(pipeline: Any) -> list[Stat]:
    """Dataset size and model, as shown alongside every page."""
    stats = []
    if pipeline.profiles_df is not None:
        stats.append(Stat("Faculty & Researchers", len(pipeline.profiles_df)))
    if pipeline.papers_df is not None:
        stats.append(Stat("Indexed Academic Papers", len(pipeline.papers_df)))
    stats.append(Stat("Embedding Model", pipeline.model_name))
    return stats


def evaluation_table(key: str) -> EvaluationTable | None:
    return next((table for table in EVALUATION_TABLES if table.key == key), None)


def load_table(table: EvaluationTable) -> TableData | None:
    """Read a metrics CSV as strings; ``None`` when the file does not exist."""
    if not table.path.exists():
        return None
    frame = pd.read_csv(table.path, dtype=str, keep_default_na=False)
    return TableData(table, tuple(frame.columns), tuple(tuple(row) for row in frame.itertuples(index=False)))


@dataclass
class PipelineLoader:
    """Loads the retrieval pipeline once and shares it between requests.

    Loading takes tens of seconds, so it can run in a background thread while
    the interface reports progress. Searches are serialised with a lock
    because the embedding model is not guaranteed to be thread-safe.
    """

    factory: Callable[[], Any]
    pipeline: Any = None
    error: str | None = None
    _started: bool = False
    _ready: threading.Event = field(default_factory=threading.Event)
    _start_lock: threading.Lock = field(default_factory=threading.Lock)
    search_lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self, background: bool = True) -> None:
        """Begin loading once; later calls do nothing."""
        with self._start_lock:
            if self._started:
                return
            self._started = True
        if background:
            threading.Thread(target=self._load, name="pipeline-loader", daemon=True).start()
        else:
            self._load()

    def _load(self) -> None:
        try:
            self.pipeline = self.factory()
        except Exception as exc:  # noqa: BLE001
            # Deliberately broad. Real load failures include classes that derive
            # directly from Exception (corrupt .npz cache: BadZipFile; corrupt model
            # weights: SafetensorError). An uncaught error would end this thread with
            # no error recorded, the app would report "ready" without a pipeline,
            # and every page would fail. The message is shown on the error page.
            self.error = str(exc) or exc.__class__.__name__
        finally:
            self._ready.set()

    @property
    def status(self) -> str:
        """``loading``, ``ready`` or ``failed``."""
        if not self._ready.is_set():
            return "loading"
        return "failed" if self.error is not None else "ready"
