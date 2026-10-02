"""Persistent local vector index for paper embeddings, backed by ChromaDB.

This is the only Week 4 module that knows ChromaDB exists. Inputs and outputs
are Week 4 objects (``Paper``, ``PaperHit``) or plain Python values; no
ChromaDB client, collection or result object is ever returned to callers.
``chromadb`` is imported lazily so this module can be imported before the
dependency is installed.

Reserved metadata namespace
---------------------------
Every key written by Week 4 itself starts with ``__`` so it can never collide
with keys coming from Week 2 via ``Paper.metadata``. Keys from
``Paper.metadata`` that start with ``__`` are rejected.

* Per-paper record keys: ``__title``, ``__abstract``, ``__authors`` (JSON).
  The paper ID is the ChromaDB record ID, so it needs no metadata key.
* Collection-level keys: ``__embedding_model`` and ``__distance_metric``
  (required), plus ``__created_at`` and ``__schema_version`` (reserved;
  optional in Phase 3).
"""

# TODO(future): other vector databases (FAISS, Qdrant, Milvus, etc.) should be
# implementable behind this same public interface without changing Retriever
# or Aggregator.

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.week4.config import DistanceMetric
from src.week4.schemas import MetadataValue, Paper, PaperHit

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

RESERVED_KEY_PREFIX = "__"


class VectorStore:
    """A ChromaDB collection of paper embeddings stored on local disk.

    The collection records which embedding model built it. Both indexing and
    searching are refused if that model differs from the one this store was
    created with, which prevents mixed-model indexes and stale searches.

    Similarity contract: every score returned by this class satisfies
    "higher score = better match". This never changes, whatever the distance
    metric or vector database, so Retriever, Aggregator and RankingStrategy
    stay backend-agnostic.

    Error messages must be actionable, e.g. "Collection not found. Run
    build_chromadb.py first." rather than a generic failure.
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        *,
        distance_metric: DistanceMetric,
        embedding_model_name: str,
    ) -> None:
        """Create a store without touching the disk.

        Args:
            persist_dir: Directory holding the ChromaDB files. Created on
                first use if missing.
            collection_name: Name of the collection holding paper embeddings.
            distance_metric: Similarity space used when the collection is created.
            embedding_model_name: Model that produced (or will produce) the
                stored embeddings; normally ``Embedder.model_name``.

        Raises:
            ValueError: If ``collection_name`` or ``embedding_model_name`` is empty.
        """
        # TODO(Phase 3): validate arguments.
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._distance_metric = distance_metric
        self._embedding_model_name = embedding_model_name
        self._collection: Collection | None = None

    def add_papers(self, papers: Sequence[Paper], embeddings: np.ndarray) -> None:
        """Store a batch of papers with their embeddings.

        Writes are upserts keyed by ``paper_id``, so rebuilding the index is
        idempotent: running ``build_chromadb.py`` multiple times never creates
        duplicate documents.

        Args:
            papers: Papers to store.
            embeddings: Array of shape ``(len(papers), embedding_dim)``; row
                ``i`` is the embedding of ``papers[i]``.

        Raises:
            ValueError: If ``papers`` is empty, its length differs from the
                number of embedding rows, it contains duplicate ``paper_id``
                values, or a ``Paper.metadata`` key uses the reserved prefix.
            RuntimeError: If ``chromadb`` is not installed or the existing
                collection was built with a different embedding model.
        """
        # TODO(Phase 3): validate, open/create the collection,
        # _check_model_compatibility, convert each paper with _to_record, upsert.
        raise NotImplementedError

    def query(self, embedding: np.ndarray, *, top_k: int) -> list[PaperHit]:
        """Find the papers most similar to a query embedding.

        Args:
            embedding: Query vector of shape ``(embedding_dim,)``.
            top_k: Maximum number of papers to return.

        Returns:
            Up to ``top_k`` hits ordered best-first. Every score follows the
            similarity contract: higher score = better match. Fewer than
            ``top_k`` if the collection holds fewer papers.

        Raises:
            ValueError: If ``top_k`` is less than 1.
            RuntimeError: If ``chromadb`` is not installed, the collection does
                not exist or is empty ("Run build_chromadb.py first."), or it
                was built with a different embedding model.
        """
        # TODO(Phase 3): open collection, _check_model_compatibility, query,
        # convert each result with _from_record and _to_similarity.
        raise NotImplementedError

    def count(self) -> int:
        """Return the number of papers stored, or 0 if the collection does not exist."""
        # TODO(Phase 3)
        raise NotImplementedError

    def metadata(self) -> dict[str, MetadataValue]:
        """Return a copy of the collection-level metadata.

        Contains at minimum ``__embedding_model`` and ``__distance_metric``;
        ``__created_at`` and ``__schema_version`` are reserved and may be present.

        Raises:
            RuntimeError: If the collection does not exist.
        """
        # TODO(Phase 3)
        raise NotImplementedError

    def _get_collection(self, *, create: bool) -> Collection:
        """Open the collection, creating it first if ``create`` is true.

        Imports ``chromadb`` lazily and uses a persistent client rooted at
        ``persist_dir``. On creation, writes the collection-level metadata
        described in the module docstring.

        Raises:
            RuntimeError: If ``chromadb`` is not installed, or the collection
                does not exist and ``create`` is false.
        """
        # TODO(Phase 2): confirm how the installed chromadb version accepts the
        # distance metric at creation (metadata key vs. configuration argument).
        # TODO(Phase 3): implement; cache the collection in self._collection.
        raise NotImplementedError

    def _check_model_compatibility(self, collection_metadata: Mapping[str, Any]) -> None:
        """Ensure the collection was built with ``embedding_model_name``.

        Raises:
            RuntimeError: If the stored model differs, naming both models and
                instructing the user to rebuild the index.
        """
        # TODO(Phase 3): compare __embedding_model with self._embedding_model_name.
        raise NotImplementedError

    @staticmethod
    def _to_record(paper: Paper) -> tuple[str, str, dict[str, MetadataValue]]:
        """Convert a ``Paper`` into a ChromaDB ``(id, document, metadata)`` record.

        The document is ``Paper.text``. Title, abstract and authors go under
        reserved keys; authors are serialized as JSON because ChromaDB metadata
        values must be scalars. ``None`` values (e.g. a missing abstract) are
        omitted.

        Raises:
            ValueError: If a ``Paper.metadata`` key uses the reserved prefix.
        """
        # TODO(Phase 3): implement with JSON author serialization.
        raise NotImplementedError

    @staticmethod
    def _from_record(paper_id: str, document: str, metadata: Mapping[str, Any]) -> Paper:
        """Rebuild a ``Paper`` from a stored ChromaDB record (inverse of ``_to_record``)."""
        # TODO(Phase 3)
        raise NotImplementedError

    def _to_similarity(self, distance: float) -> float:
        """Convert a ChromaDB distance into a similarity where higher is better.

        Because every embedding is L2-normalized, each metric maps to cosine
        similarity:

        * ``cosine``: similarity = 1 - distance
        * ``ip``: similarity = 1 - distance
        * ``l2`` (ChromaDB reports squared L2): similarity = 1 - distance / 2
        """
        # TODO(Phase 3)
        raise NotImplementedError
