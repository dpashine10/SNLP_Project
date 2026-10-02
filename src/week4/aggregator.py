"""Turns paper-level search hits into a ranked list of researchers.

The aggregator coordinates; it never scores. It groups hits by author, asks
the injected ranking strategy for each author's score, then orders and
truncates the result. Changing the strategy changes the ranking without
changing this module.

It never performs retrieval, generates embeddings, accesses ChromaDB, reads
files or reads configuration. Everything it needs is passed in.
"""

from collections.abc import Sequence

from src.week4.ranking_strategy import RankingStrategy
from src.week4.schemas import Author, PaperHit, ResearcherScore


# TODO(future): aggregation levels may include institution,
# department or laboratory. The current implementation is researcher-level only.
class Aggregator:
    """Groups paper hits by researcher and ranks researchers with a strategy."""

    def __init__(self, strategy: RankingStrategy) -> None:
        """Create an aggregator.

        Args:
            strategy: Scoring strategy, obtained from ``create_strategy``.
        """
        self._strategy = strategy

    def aggregate(self, hits: Sequence[PaperHit], *, top_k: int) -> list[ResearcherScore]:
        """Rank researchers from the paper hits retrieved for one query.

        Processing pipeline::

            PaperHits
              -> group by author        (_group_by_author)
              -> compute researcher score via the strategy (_score_author)
              -> sort researchers       (_rank)
              -> return top_k

        A paper with several authors credits its hit to each of them. Each
        ``PaperHit`` is processed exactly once before researchers are sorted,
        so cost grows linearly with the number of hits plus one sort over the
        researchers found.

        Ordering is deterministic:

        * Researchers are ordered by descending score, then ascending ``author_id``.
        * Evidence papers are ordered by descending similarity score, then
          ascending ``paper_id``.

        Every researcher keeps all of their retrieved hits as evidence; only
        the list of researchers is truncated to ``top_k``.

        Args:
            hits: Paper hits for a single query, in any order.
            top_k: Maximum number of researchers to return.

        Returns:
            At most ``top_k`` researchers ordered from highest to lowest score.
            An empty list if ``hits`` is empty.

        Raises:
            ValueError: If ``top_k`` is less than 1.
        """
        # TODO(Phase 3): validate top_k, then _group_by_author -> _score_author
        # for each group -> _rank.
        raise NotImplementedError

    @staticmethod
    def _group_by_author(hits: Sequence[PaperHit]) -> dict[str, tuple[Author, list[PaperHit]]]:
        """Group hits by ``author_id``.

        Returns:
            A mapping from ``author_id`` to the first ``Author`` object seen
            for that ID and every hit credited to that author.
        """
        # TODO(Phase 3)
        raise NotImplementedError

    def _score_author(self, author: Author, hits: Sequence[PaperHit]) -> ResearcherScore:
        """Score one researcher with the strategy and package the result.

        Evidence is all of the researcher's hits, ordered by descending score
        then ascending ``paper_id``. ``matched_papers`` is the number of hits
        received, i.e. the papers retrieved for this researcher before scoring.
        """
        # TODO(Phase 3)
        raise NotImplementedError

    @staticmethod
    def _rank(scores: Sequence[ResearcherScore], top_k: int) -> list[ResearcherScore]:
        """Order by descending score, then ascending ``author_id``, and keep ``top_k``."""
        # TODO(Phase 3)
        raise NotImplementedError
