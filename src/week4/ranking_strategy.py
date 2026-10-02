"""Strategies that turn one researcher's paper hits into a single score.

Each strategy answers one question: given every retrieved paper by a
researcher, how relevant is that researcher? ``aggregator`` groups hits by
author and delegates scoring to the strategy selected in ``config``, so
strategies can be swapped for experiments without touching other modules.

Strategy objects are constructed only through ``create_strategy``.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import ClassVar

from src.week4.config import RankingStrategyName, Week4Config
from src.week4.schemas import PaperHit


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
        # TODO(Phase 3): return the highest hit score.
        raise NotImplementedError


class AverageStrategy(RankingStrategy):
    """Scores a researcher by the mean of all their retrieved hits.

    Rewards consistent relevance; a single strong paper is diluted by weaker ones.
    """

    name = "average"

    def score(self, hits: Sequence[PaperHit]) -> float:
        # TODO(Phase 3): return the mean hit score.
        raise NotImplementedError


class TopKAverageStrategy(RankingStrategy):
    """Scores a researcher by the mean of their ``k`` best hits.

    Uses all hits when the researcher has fewer than ``k``. Balances
    specialists against researchers with several relevant papers.
    """

    name = "top_k_average"

    def __init__(self, k: int) -> None:
        """Create the strategy.

        Args:
            k: Number of best hits to average.

        Raises:
            ValueError: If ``k`` is less than 1.
        """
        # TODO(Phase 3): validate k.
        self._k = k

    def score(self, hits: Sequence[PaperHit]) -> float:
        # TODO(Phase 3): return the mean of the k highest hit scores.
        raise NotImplementedError


# TODO(Phase 3): future weighting may include publication recency, citation
# count, venue quality and configurable weighting coefficients. Any such
# signal must reach the strategy through PaperHit/Paper metadata, never by
# the strategy fetching data itself.
class WeightedStrategy(RankingStrategy):
    """Placeholder for a weighted combination of hit scores (formula not decided)."""

    name = "weighted"

    def score(self, hits: Sequence[PaperHit]) -> float:
        # TODO(Phase 3): implement once the weighting formula is decided.
        raise NotImplementedError


_StrategyBuilder = Callable[[Week4Config], RankingStrategy]

# To add a strategy, define its class above, add its name to
# RankingStrategyName in config, and register a builder here.
_STRATEGY_REGISTRY: dict[RankingStrategyName, _StrategyBuilder] = {
    "best_paper": lambda config: BestPaperStrategy(),
    "average": lambda config: AverageStrategy(),
    "top_k_average": lambda config: TopKAverageStrategy(k=config.strategy_top_k),
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
    # TODO(Phase 3): look up config.ranking_strategy in _STRATEGY_REGISTRY,
    # raise ValueError listing the supported names if missing, otherwise call
    # the builder with config.
    raise NotImplementedError
