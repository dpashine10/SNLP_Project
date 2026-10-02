"""Contracts between Week 4 modules and their interchangeable implementations.

Only two seams need a protocol:

* ``PaperSource``: the contract indexing needs from a paper supplier, so
  ``build_index`` never depends on Week 2 (implemented by ``data_adapter``).
* ``ResearcherRecommender``: ranks researchers for a query (implemented by
  ``baseline_adapter`` and ``week4_pipeline``), letting ``evaluation`` compare
  the baseline and the improved model through identical code.
"""

from collections.abc import Iterable, Sequence
from typing import Protocol

from src.week4.schemas import Paper, ResearcherScore


class PaperSource(Protocol):
    """A source of papers to be embedded and indexed."""

    def load_papers(self) -> Iterable[Paper]:
        """Yield every paper available for indexing.

        Returned lazily so large datasets can be indexed in batches without
        holding every paper in memory.

        Returns:
            An iterable of ``Paper`` objects with unique ``paper_id`` values.

        Raises:
            FileNotFoundError: If the underlying source data is missing.
            ValueError: If the source data cannot be mapped to ``Paper``.
        """
        ...


class ResearcherRecommender(Protocol):
    """A system that ranks researchers by relevance to a free-text query."""

    @property
    def name(self) -> str:
        """Short human-readable label used in evaluation output."""
        ...

    def recommend(self, query: str, *, top_k: int) -> Sequence[ResearcherScore]:
        """Rank researchers for a query.

        The returned sequence must already be ordered from highest relevance
        to lowest relevance.

        Args:
            query: Research topic, problem description or abstract.
            top_k: Maximum number of researchers to return.

        Returns:
            At most ``top_k`` results, ordered best-first.

        Raises:
            ValueError: If ``query`` is empty or ``top_k`` is less than 1.
        """
        ...
