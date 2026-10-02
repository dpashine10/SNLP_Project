"""The complete Week 4 researcher recommender.

Orchestration only::

    query -> Retriever -> PaperHits -> Aggregator -> ResearcherScores

``Week4Pipeline`` satisfies the ``ResearcherRecommender`` protocol, so
``evaluation`` can compare it with the Week 3 baseline through identical
code. It never accesses ChromaDB, computes embeddings or similarities,
groups authors, ranks researchers or reads files; those belong to the
injected components. Errors from those components propagate unchanged.

``build_week4_pipeline`` is the composition root for querying: entry points
obtain pipelines through it rather than constructing components themselves.
"""

# TODO(future): query preprocessing, caching, request tracing, telemetry and
# performance monitoring belong around this orchestration layer, not inside
# Retriever or Aggregator.

import logging

from src.week4.aggregator import Aggregator
from src.week4.config import Week4Config
from src.week4.embedder import Embedder
from src.week4.ranking_strategy import create_strategy
from src.week4.retriever import Retriever
from src.week4.schemas import ResearcherScore
from src.week4.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Week4Pipeline:
    """Recommends researchers via paper-level semantic search and aggregation."""

    def __init__(
        self,
        retriever: Retriever,
        aggregator: Aggregator,
        *,
        paper_top_k: int,
        name: str,
    ) -> None:
        """Create the pipeline from already-built components.

        Args:
            retriever: Finds papers relevant to a query.
            aggregator: Ranks researchers from paper hits.
            paper_top_k: Papers retrieved per query before aggregation.
            name: Label used in evaluation output, e.g. "Week 4 (best_paper)".

        Raises:
            ValueError: If ``paper_top_k`` is less than 1 or ``name`` is empty.
        """
        if paper_top_k < 1:
            raise ValueError(f"paper_top_k must be at least 1, got {paper_top_k}.")
        if not name.strip():
            raise ValueError("name must not be empty.")
        self._retriever = retriever
        self._aggregator = aggregator
        self._paper_top_k = paper_top_k
        self._name = name

    @property
    def name(self) -> str:
        """Short human-readable label used in evaluation output."""
        return self._name

    def recommend(self, query: str, *, top_k: int) -> list[ResearcherScore]:
        """Rank researchers for a query.

        The returned sequence is ordered from highest relevance to lowest
        relevance.

        Args:
            query: Research topic, problem description or abstract.
            top_k: Maximum number of researchers to return.

        Returns:
            At most ``top_k`` researchers, best-first, each carrying their
            paper hits as evidence.

        Raises:
            ValueError: If ``query`` is blank or ``top_k`` is less than 1
                (raised by the retriever or aggregator, not re-checked here).
            RuntimeError: If the model cannot be loaded or the index is
                missing, empty or incompatible.
        """
        hits = self._retriever.retrieve(query, top_k=self._paper_top_k)
        researchers = self._aggregator.aggregate(hits, top_k=top_k)
        logger.debug(
            "%s: %d-character query, %d of %d papers retrieved, %d of %d researchers returned.",
            self._name,
            len(query),
            len(hits),
            self._paper_top_k,
            len(researchers),
            top_k,
        )
        return researchers


def build_week4_pipeline(config: Week4Config) -> Week4Pipeline:
    """Build a ready-to-use pipeline from configuration.

    The pipeline is labelled with the active ranking strategy. No model or
    index is loaded until the first query.

    Args:
        config: Settings for every component.

    Returns:
        A ``Week4Pipeline`` named after ``config.ranking_strategy``.

    Raises:
        ValueError: If any configured value is invalid (raised by the
            component that owns it).
    """
    # Construction order:
    #   Embedder -> VectorStore -> Retriever -> RankingStrategy -> Aggregator
    #   -> Week4Pipeline
    embedder = Embedder(
        config.embedding_model_name, device=config.device, batch_size=config.batch_size
    )
    store = VectorStore(
        config.chroma_dir,
        config.collection_name,
        distance_metric=config.distance_metric,
        embedding_model_name=embedder.model_name,
    )
    retriever = Retriever(embedder, store)
    aggregator = Aggregator(create_strategy(config))
    return Week4Pipeline(
        retriever,
        aggregator,
        paper_top_k=config.paper_top_k,
        name=f"Week 4 ({config.ranking_strategy})",
    )
