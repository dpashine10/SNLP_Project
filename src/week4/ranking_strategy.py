"""Strategies that turn one researcher's paper hits into a single score.

Each strategy answers one question: given every retrieved paper by a
researcher, how relevant is that researcher? ``aggregator`` groups hits by
author and delegates scoring to the strategy selected in ``config``, so
strategies can be swapped for experiments without touching other modules.

Strategy objects are constructed only through ``create_strategy``.
"""

import heapq
import math
import statistics
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import ClassVar

from src.week4.config import RankingStrategyName, Week4Config
from src.week4.schemas import PaperHit


def _hit_scores(hits: Sequence[PaperHit]) -> list[float]:
    """Return the hit scores, rejecting an empty sequence as strategies require."""
    if not hits:
        raise ValueError("A researcher needs at least one paper hit to be scored.")
    return [hit.score for hit in hits]


class RankingStrategy(ABC):
    """Base class for researcher scoring strategies.

    A strategy is a pure scoring function. It only computes one score from
    the hits it is given; it never groups authors, filters papers, sorts
    researchers or performs retrieval. It must be deterministic: the same
    hits always produce the same score, with no randomness involved.

    Scores are comparable between researchers ranked by the same strategy,
    but not across different strategies.
    """

    name: ClassVar[RankingStrategyName]

    @abstractmethod
    def score(self, hits: Sequence[PaperHit]) -> float:
        """Combine one researcher's paper hits into a single relevance score.

        Args:
            hits: Every retrieved paper hit for a single researcher, in any
                order. Each ``PaperHit.score`` is already a normalized
                similarity where higher means a better match; strategies never
                need to know which distance metric the vector store used.

        Returns:
            The researcher's relevance score; higher means more relevant.

        Raises:
            ValueError: If ``hits`` is empty.
        """


class BestPaperStrategy(RankingStrategy):
    """Scores a researcher by their single best-matching paper.

    Favours specialists: one highly relevant paper is enough to rank highly.
    """

    name = "best_paper"

    def score(self, hits: Sequence[PaperHit]) -> float:
        return max(_hit_scores(hits))


class AverageStrategy(RankingStrategy):
    """Scores a researcher by the mean of all their retrieved hits.

    Rewards consistent relevance; a single strong paper is diluted by weaker ones.
    """

    name = "average"

    def score(self, hits: Sequence[PaperHit]) -> float:
        return statistics.fmean(_hit_scores(hits))


class TopKAverageStrategy(RankingStrategy):
    """Scores a researcher by the mean of their ``k`` best hits.

    Uses all hits when the researcher has fewer than ``k``. Balances
    specialists against researchers with several relevant papers.
    """

    # Annotated so the author-discount subclass may use its own name.
    name: ClassVar[RankingStrategyName] = "top_k_average"

    def __init__(self, k: int) -> None:
        """Create the strategy.

        Args:
            k: Number of best hits to average.

        Raises:
            ValueError: If ``k`` is less than 1.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}.")
        self._k = k

    def score(self, hits: Sequence[PaperHit]) -> float:
        return statistics.fmean(heapq.nlargest(self._k, _hit_scores(hits)))


class TopKAverageAuthorDiscountStrategy(TopKAverageStrategy):
    """Top-k average after discounting each hit by its paper's author count.

    A paper with ``n`` authors contributes ``score / sqrt(n)`` to each of its
    co-authors, so one large-team paper no longer lifts many co-authors to a
    high score; single-author papers are unchanged. Co-authors of the same
    paper still receive equal credit for it. Added in Week 5 after the error
    analysis found co-author tie inflation.
    """

    name = "top_k_average_author_discount"

    def score(self, hits: Sequence[PaperHit]) -> float:
        discounted = [
            score / math.sqrt(max(len(hit.paper.authors), 1))
            for score, hit in zip(_hit_scores(hits), hits, strict=True)
        ]
        return statistics.fmean(heapq.nlargest(self._k, discounted))


# TODO(future): weighting may include publication recency, citation
# count, venue quality and configurable weighting coefficients. Any such
# signal must reach the strategy through PaperHit/Paper metadata, never by
# the strategy fetching data itself.
class WeightedStrategy(RankingStrategy):
    """Placeholder for a weighted combination of hit scores (formula not decided)."""

    name = "weighted"

    def score(self, hits: Sequence[PaperHit]) -> float:
        # TODO(future): implement once the weighting formula is decided.
        raise NotImplementedError


_StrategyBuilder = Callable[[Week4Config], RankingStrategy]

# To add a strategy, define its class above, add its name to
# RankingStrategyName in config, and register a builder here.
_STRATEGY_REGISTRY: dict[RankingStrategyName, _StrategyBuilder] = {
    "best_paper": lambda config: BestPaperStrategy(),
    "average": lambda config: AverageStrategy(),
    "top_k_average": lambda config: TopKAverageStrategy(k=config.strategy_top_k),
    "top_k_average_author_discount": lambda config: TopKAverageAuthorDiscountStrategy(
        k=config.strategy_top_k
    ),
    "weighted": lambda config: WeightedStrategy(),
}


def create_strategy(config: Week4Config) -> RankingStrategy:
    """Build the strategy named by ``config.ranking_strategy``.

    Args:
        config: Settings providing the strategy name and its parameters.

    Returns:
        A ready-to-use ranking strategy.

    Raises:
        ValueError: If ``config.ranking_strategy`` is not a registered strategy.
    """
    builder = _STRATEGY_REGISTRY.get(config.ranking_strategy)
    if builder is None:
        supported = ", ".join(_STRATEGY_REGISTRY)
        raise ValueError(
            f"Unknown ranking strategy {config.ranking_strategy!r}; supported: {supported}."
        )
    return builder(config)
