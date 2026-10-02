"""Paper-level semantic retrieval: query text in, paper hits out.

The retriever is an orchestration layer only::

    query -> Embedder.embed_query() -> VectorStore.query() -> PaperHits

It never computes or normalizes similarities, accesses ChromaDB APIs or CSV
files, knows about Week 2 or Week 3, applies ranking strategies or
aggregates researchers. Validation is owned by the components it calls
(Embedder for queries, VectorStore for the index); errors are propagated
unchanged.
"""

# TODO(future): retrieval improvements such as hybrid lexical + semantic
# retrieval, reranking or cross-encoder reranking belong after VectorStore
# retrieval and before Aggregator.

import logging

from src.week4.embedder import Embedder
from src.week4.schemas import PaperHit
from src.week4.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Finds the papers most relevant to a free-text query."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        """Create a retriever.

        Args:
            embedder: Embedder used for queries. Must use the same model that
                built the index; ``VectorStore`` enforces this on search.
            store: Vector store holding the indexed papers.
        """
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, *, top_k: int) -> list[PaperHit]:
        """Return the papers most similar to ``query``.

        Args:
            query: Research topic, problem description or abstract.
            top_k: Maximum number of papers to return.

        Returns:
            Up to ``top_k`` hits ordered best-first, scored by the vector
            store's similarity contract (higher score = better match).

        Raises:
            ValueError: If ``query`` is blank or ``top_k`` is less than 1
                (raised by the embedder or vector store, not re-checked here).
            RuntimeError: If the model cannot be loaded or the index is
                missing, empty or built with a different embedding model.
        """
        # TODO(future): optional metadata filters (publication year,
        # institution, department, author, venue) would be accepted here as
        # plain Python values; only VectorStore translates them into
        # backend-specific filter syntax.
        embedding = self._embedder.embed_query(query)
        hits = self._store.query(embedding, top_k=top_k)
        logger.debug(
            "Retrieved %d of %d requested papers for a %d-character query (embedding dimension %d).",
            len(hits),
            top_k,
            len(query),
            embedding.shape[0],
        )
        return hits
