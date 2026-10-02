"""Central configuration for Week 4.

Every path, model name and tunable value used by Week 4 is defined here.
Other modules receive a ``Week4Config`` instance rather than declaring their
own constants, so an experiment only requires changing one value.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

RankingStrategyName = Literal["best_paper", "average", "top_k_average", "weighted"]
DeviceName = Literal["auto", "cpu", "cuda"]
DistanceMetric = Literal["cosine", "l2", "ip"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

SUPPORTED_RANKING_STRATEGIES: tuple[str, ...] = get_args(RankingStrategyName)
SUPPORTED_DEVICES: tuple[str, ...] = get_args(DeviceName)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
WEEK4_DATA_DIR: Path = PROJECT_ROOT / "Data" / "week4"


@dataclass(frozen=True)
class Week4Config:
    """Immutable settings for the Week 4 pipeline.

    Attributes:
        papers_path: Week 2 paper-level output. Read only by ``data_adapter``.
        authorship_path: Week 2 paper-author relationships, if stored separately
            from the papers. Read only by ``data_adapter``.
        evaluation_labels_path: Labelled evaluation queries. Read only by the
            label loader behind ``evaluation.load_labeled_queries``.
        chroma_dir: Directory where the persistent ChromaDB index is stored.
        results_dir: Directory where evaluation outputs are written.
        collection_name: ChromaDB collection holding paper embeddings.
        distance_metric: Similarity space used by the ChromaDB index.
        embedding_model_name: Sentence Transformers model for papers and queries.
        device: Inference device; ``"auto"`` lets the embedder pick one.
        batch_size: Number of papers encoded and inserted per indexing batch.
        paper_top_k: Papers retrieved from ChromaDB per query, before
            aggregation into researchers.
        researcher_top_k: Researchers returned to the caller.
        ranking_strategy: Strategy used to turn a researcher's paper hits
            into a single score.
        strategy_top_k: Number of best papers averaged by ``top_k_average``.
        random_seed: Seed for any stochastic step, for reproducible runs.
        log_level: Logging verbosity for Week 4 modules.
    """

    # -------------------------
    # Phase 2 Inputs
    # -------------------------
    # TODO(Phase 2): set once the final Week 2 output files are known.
    papers_path: Path | None = None
    authorship_path: Path | None = None

    # TODO(Phase 2): set once the evaluation label source is decided.
    evaluation_labels_path: Path | None = None

    # -------------------------
    # Storage
    # -------------------------
    chroma_dir: Path = WEEK4_DATA_DIR / "chroma"
    results_dir: Path = WEEK4_DATA_DIR / "results"
    collection_name: str = "iitb_research_papers"
    distance_metric: DistanceMetric = "cosine"

    # -------------------------
    # Embedding
    # -------------------------
    # Other models may be evaluated in future experiments, but the default stays
    # identical to the Week 3 baseline so the comparison remains fair.
    embedding_model_name: str = "all-MiniLM-L6-v2"
    device: DeviceName = "auto"
    batch_size: int = 256

    # -------------------------
    # Retrieval
    # -------------------------
    paper_top_k: int = 100
    # Not wired yet: reserved as the default top_k for a future CLI/API caller.
    researcher_top_k: int = 10

    # -------------------------
    # Ranking
    # -------------------------
    ranking_strategy: RankingStrategyName = "best_paper"
    strategy_top_k: int = 3
    # TODO(Phase 3): add parameters for the "weighted" strategy once its formula is decided.

    # -------------------------
    # Evaluation
    # -------------------------
    random_seed: int = 42

    # -------------------------
    # Logging
    # -------------------------
    log_level: LogLevel = "INFO"


CONFIG = Week4Config()
