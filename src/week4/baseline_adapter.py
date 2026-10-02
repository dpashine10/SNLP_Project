"""Adapter that exposes the Week 3 baseline as a ``ResearcherRecommender``.

This is the ONLY Week 4 module that imports or wraps Week 3 code. All Week 3
compatibility logic (imports, call signatures, result formats, search mode
names) lives here. If Week 3 changes after a ``git pull``, only this module
should need to change.

The Week 3 baseline is ``ResearchDiscoveryPipeline`` in
``src/week2/pipeline.py``, which searches the researcher embeddings built by
``src/week3/build_embeddings.py``. It is imported lazily, so importing this
module never loads Week 3 code or its models, and one loaded instance is
shared by every mode.

``semantic`` is the default mode because it uses the same embedding model as
Week 4, so comparing the two isolates the architectural change.
"""

# TODO(future): other baselines (e.g. BM25, FAISS, external APIs) would be
# separate ResearcherRecommender implementations in their own modules, so
# this file stays Week 3 only and evaluation.py needs no change.

import functools
import importlib
import logging
from typing import Any, Literal, get_args

from src.week4.schemas import Author, ResearcherScore

logger = logging.getLogger(__name__)

BaselineMode = Literal["semantic", "tfidf", "hybrid"]
PRIMARY_BASELINE_MODE: BaselineMode = "semantic"

_BASELINE_MODULE = "src.week2.pipeline"


class Week3Baseline:
    """The Week 3 recommender in a given search mode.

    Satisfies the ``ResearcherRecommender`` protocol.

    Scores produced here are on Week 3's scale and must not be compared
    numerically with Week 4 scores; evaluation compares rankings only.
    """

    def __init__(self, *, mode: BaselineMode = PRIMARY_BASELINE_MODE) -> None:
        """Create the adapter without loading Week 3.

        ``hybrid`` uses Week 3's own default blend weight.

        Args:
            mode: Week 3 search mode to evaluate.

        Raises:
            ValueError: If ``mode`` is not a supported baseline mode.
        """
        if mode not in get_args(BaselineMode):
            raise ValueError(
                f"Unsupported baseline mode {mode!r}; expected one of "
                f"{', '.join(get_args(BaselineMode))}."
            )
        self._mode = mode

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
            At most ``top_k`` researchers, best-first, ties broken by
            ``author_id``. ``evidence`` is empty and ``matched_papers`` is 0,
            because Week 3 does not expose the paper hits behind its scores
            in Week 4 form.

        Raises:
            ValueError: If ``query`` is blank or ``top_k`` is less than 1.
            RuntimeError: If the Week 3 code cannot be imported or initialized,
                or returns output in an unexpected format.
        """
        if not query.strip():
            raise ValueError("query must not be blank.")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")
        raw_results = self._get_engine().search(query=query, top_k=top_k, mode=self._mode)
        results = [self._to_researcher_score(raw_result) for raw_result in raw_results]
        results.sort(key=lambda result: (-result.score, result.author.author_id))
        return results[:top_k]

    @staticmethod
    @functools.cache
    def _get_engine() -> Any:
        """Import and initialize the Week 3 recommender on first use.

        The instance is shared by every ``Week3Baseline``, because Week 3
        takes the search mode per call and loading it is slow (it reads every
        profile and fits TF-IDF).

        Raises:
            RuntimeError: If the Week 3 code is missing or fails to
                initialize, with a message naming the module that failed.
        """
        try:
            engine = importlib.import_module(_BASELINE_MODULE).ResearchDiscoveryPipeline()
        except Exception as error:
            raise RuntimeError(
                f"The Week 3 baseline ({_BASELINE_MODULE}) failed to load: {error}"
            ) from error
        logger.info("Loaded the Week 3 baseline from %s.", _BASELINE_MODULE)
        return engine

    @staticmethod
    def _to_researcher_score(raw_result: Any) -> ResearcherScore:
        """Convert one Week 3 result into a ``ResearcherScore``.

        Week 3 returns the same OpenAlex author IDs as Week 2's data and the
        evaluation labels (all come from Week 2's processed files), so the
        IDs are already canonical and used as-is.

        Raises:
            RuntimeError: If ``raw_result`` lacks the expected fields.
        """
        try:
            author = Author(
                author_id=str(raw_result["author_id"]).strip(),
                name=str(raw_result["author_name"]).strip(),
            )
            return ResearcherScore(author=author, score=float(raw_result["similarity_score"]))
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Unexpected Week 3 result format: {raw_result!r}") from error
