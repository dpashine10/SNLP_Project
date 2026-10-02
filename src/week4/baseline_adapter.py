"""Adapter that exposes the Week 3 baseline as a ``ResearcherRecommender``.

This is the ONLY Week 4 module that imports or wraps Week 3 code. All Week 3
compatibility logic (imports, call signatures, result formats, search mode
names) lives here. If Week 3 changes after a ``git pull``, only this module
should need to change.

Week 3 is imported lazily, so importing this module never loads Week 3 code
or its models.

``semantic`` is the default mode because it uses the same embedding model as
Week 4, so comparing the two isolates the architectural change.
"""

# TODO(future): other baselines (e.g. BM25, FAISS, external APIs) would be
# separate ResearcherRecommender implementations in their own modules, so
# this file stays Week 3 only and evaluation.py needs no change.

import logging
from typing import Any, Literal

from src.week4.schemas import ResearcherScore

logger = logging.getLogger(__name__)

BaselineMode = Literal["semantic", "tfidf", "hybrid"]
PRIMARY_BASELINE_MODE: BaselineMode = "semantic"


class Week3Baseline:
    """The Week 3 recommender in a given search mode.

    Satisfies the ``ResearcherRecommender`` protocol.

    Scores produced here are on Week 3's scale and must not be compared
    numerically with Week 4 scores; evaluation compares rankings only.
    """

    def __init__(self, *, mode: BaselineMode = PRIMARY_BASELINE_MODE) -> None:
        """Create the adapter without loading Week 3.

        Args:
            mode: Week 3 search mode to evaluate.

        Raises:
            ValueError: If ``mode`` is not a supported baseline mode.
        """
        # TODO(Phase 3): validate mode.
        # TODO(Phase 2): if a mode needs extra parameters (e.g. a hybrid blend
        # weight), add them here once the final Week 3 API is known.
        self._mode = mode
        self._engine: Any | None = None

    @property
    def name(self) -> str:
        """Short human-readable label used in evaluation output, e.g. "Week 3 (semantic)"."""
        return f"Week 3 ({self._mode})"

    def recommend(self, query: str, *, top_k: int) -> list[ResearcherScore]:
        """Rank researchers for a query using the Week 3 baseline.

        The returned sequence is ordered from highest relevance to lowest
        relevance. Week 3 output is not trusted to follow the Week 4
        contract, so this adapter enforces it: inputs are validated here and
        results are re-ordered best-first and truncated to ``top_k``.

        Args:
            query: Research topic, problem description or abstract.
            top_k: Maximum number of researchers to return.

        Returns:
            At most ``top_k`` researchers, best-first. ``evidence`` is empty
            and ``matched_papers`` is 0, because Week 3 does not expose the
            paper hits behind its scores in Week 4 form.

        Raises:
            ValueError: If ``query`` is blank or ``top_k`` is less than 1.
            RuntimeError: If the Week 3 code cannot be imported or initialized,
                or returns output in an unexpected format.
        """
        # TODO(Phase 3): validate query and top_k (Week 3 may not enforce the
        # Week 4 contract itself, so this is not duplicate validation).
        # TODO(Phase 2): call the Week 3 engine in self._mode, convert each
        # result with _to_researcher_score, sort best-first, truncate to top_k.
        raise NotImplementedError

    def _get_engine(self) -> Any:
        """Import and initialize the Week 3 recommender on first use.

        Raises:
            RuntimeError: If the Week 3 code is missing or fails to
                initialize, with a message naming the module that failed.
        """
        # TODO(Phase 2): lazily import the final Week 3 entry point after
        # pulling teammate code; cache it in self._engine.
        raise NotImplementedError

    @staticmethod
    def _to_researcher_score(raw_result: Any) -> ResearcherScore:
        """Convert one Week 3 result into a ``ResearcherScore``.

        Raises:
            RuntimeError: If ``raw_result`` lacks the expected fields.
        """
        # TODO(Phase 2): map the Week 3 result format to Author(author_id,
        # name) and score. Normalization must guarantee that Week 2 IDs,
        # Week 3 IDs and ground-truth label IDs all resolve to exactly the
        # same canonical researcher identifier before evaluation; otherwise
        # correct results would be scored as misses.
        raise NotImplementedError
