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
from src.week4.retriever import Retriever
from src.week4.schemas import ResearcherScore

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
        # TODO(Phase 3): validate paper_top_k and name.
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
        # TODO(Phase 3): retrieve paper_top_k hits, then aggregate to top_k
        # researchers. Log at DEBUG level: query length, paper_top_k,
        # researcher top_k, retrieved paper count, returned researcher count
        # and (optionally) pipeline latency.
        raise NotImplementedError


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
    # TODO(Phase 3): construct each component from config values and inject
    # them into Week4Pipeline.
    raise NotImplementedError
