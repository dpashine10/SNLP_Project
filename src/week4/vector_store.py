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
  (required), plus ``__created_at`` and ``__schema_version`` (reserved; not
  written yet).
"""

# TODO(future): other vector databases (FAISS, Qdrant, Milvus, etc.) should be
# implementable behind this same public interface without changing Retriever
# or Aggregator.

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.week4.config import DistanceMetric
from src.week4.schemas import Author, MetadataValue, Paper, PaperHit

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

RESERVED_KEY_PREFIX = "__"
_TITLE_KEY = "__title"
_ABSTRACT_KEY = "__abstract"
_AUTHORS_KEY = "__authors"
_MODEL_KEY = "__embedding_model"
_METRIC_KEY = "__distance_metric"


class _CollectionNotFoundError(RuntimeError):
    """The collection has not been built yet."""


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
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty.")
        if not embedding_model_name.strip():
            raise ValueError("embedding_model_name must not be empty.")
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
        if not papers:
            raise ValueError("papers must not be empty.")
        if len(papers) != len(embeddings):
            raise ValueError(f"Got {len(papers)} papers but {len(embeddings)} embedding rows.")
        records = [self._to_record(paper) for paper in papers]
        ids = [paper_id for paper_id, _, _ in records]
        if len(set(ids)) != len(ids):
            raise ValueError("papers contains duplicate paper_id values.")
        collection = self._get_collection(create=True)
        self._check_model_compatibility(collection.metadata)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=[document for _, document, _ in records],
            metadatas=[metadata for _, _, metadata in records],
        )

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
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")
        collection = self._get_collection(create=False)
        self._check_model_compatibility(collection.metadata)
        size = collection.count()
        if size == 0:
            raise RuntimeError(
                f"Collection {self._collection_name!r} is empty. Run build_chromadb.py first."
            )
        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, size),
            include=["documents", "metadatas", "distances"],
        )
        documents, metadatas, distances = (
            result["documents"],
            result["metadatas"],
            result["distances"],
        )
        if documents is None or metadatas is None or distances is None:
            raise RuntimeError("ChromaDB returned a query result without the requested fields.")
        return [
            PaperHit(
                paper=self._from_record(paper_id, str(document), metadata),
                score=self._to_similarity(distance),
            )
            for paper_id, document, metadata, distance in zip(
                result["ids"][0], documents[0], metadatas[0], distances[0], strict=True
            )
        ]

    def count(self) -> int:
        """Return the number of papers stored, or 0 if the collection does not exist."""
        try:
            return self._get_collection(create=False).count()
        except _CollectionNotFoundError:
            return 0

    def metadata(self) -> dict[str, MetadataValue]:
        """Return a copy of the collection-level metadata.

        Contains at minimum ``__embedding_model`` and ``__distance_metric``;
        ``__created_at`` and ``__schema_version`` are reserved and may be present.

        Raises:
            RuntimeError: If the collection does not exist.
        """
        # TODO(future): not needed by indexing, search or evaluation.
        raise NotImplementedError

    def _get_collection(self, *, create: bool) -> Collection:
        """Open the collection, creating it first if ``create`` is true.

        Imports ``chromadb`` lazily and uses a persistent client rooted at
        ``persist_dir``. On creation, writes the collection-level metadata
        described in the module docstring and sets the distance metric through
        the collection's HNSW configuration.

        Raises:
            RuntimeError: If ``chromadb`` is not installed, or the collection
                does not exist and ``create`` is false.
        """
        if self._collection is not None:
            return self._collection
        not_found = (
            f"Collection {self._collection_name!r} not found in {self._persist_dir}. "
            "Run build_chromadb.py first."
        )
        if not create and not self._persist_dir.exists():
            raise _CollectionNotFoundError(not_found)
        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.errors import NotFoundError
        except ImportError as error:
            raise RuntimeError(
                "chromadb is not installed; install the project dependencies with `uv sync`."
            ) from error
        client = chromadb.PersistentClient(
            path=str(self._persist_dir), settings=Settings(anonymized_telemetry=False)
        )
        if create:
            collection = client.get_or_create_collection(
                self._collection_name,
                configuration={"hnsw": {"space": self._distance_metric}},
                metadata={
                    _MODEL_KEY: self._embedding_model_name,
                    _METRIC_KEY: self._distance_metric,
                },
            )
        else:
            try:
                collection = client.get_collection(self._collection_name)
            except NotFoundError as error:
                raise _CollectionNotFoundError(not_found) from error
        self._collection = collection
        return collection

    def _check_model_compatibility(self, collection_metadata: Mapping[str, Any] | None) -> None:
        """Ensure the collection was built with this store's model and distance metric.

        Raises:
            RuntimeError: If either differs, naming both values and instructing
                the user to rebuild the index.
        """
        stored = collection_metadata or {}
        built_with = (stored.get(_MODEL_KEY), stored.get(_METRIC_KEY))
        configured = (self._embedding_model_name, self._distance_metric)
        if built_with != configured:
            raise RuntimeError(
                f"Collection {self._collection_name!r} was built with (model, metric) "
                f"{built_with}, but {configured} is configured. Delete "
                f"{self._persist_dir} and run build_chromadb.py to rebuild the index."
            )

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
        reserved = sorted(key for key in paper.metadata if key.startswith(RESERVED_KEY_PREFIX))
        if reserved:
            raise ValueError(
                f"Paper {paper.paper_id} uses reserved metadata keys: {', '.join(reserved)}."
            )
        metadata: dict[str, MetadataValue] = dict(paper.metadata)
        metadata[_TITLE_KEY] = paper.title
        metadata[_AUTHORS_KEY] = json.dumps(
            [{"author_id": author.author_id, "name": author.name} for author in paper.authors]
        )
        if paper.abstract is not None:
            metadata[_ABSTRACT_KEY] = paper.abstract
        return paper.paper_id, paper.text, metadata

    @staticmethod
    def _from_record(paper_id: str, document: str, metadata: Mapping[str, Any]) -> Paper:
        """Rebuild a ``Paper`` from a stored ChromaDB record (inverse of ``_to_record``)."""
        abstract = metadata.get(_ABSTRACT_KEY)
        return Paper(
            paper_id=paper_id,
            title=str(metadata.get(_TITLE_KEY, "")),
            text=document,
            authors=tuple(
                Author(author_id=author["author_id"], name=author["name"])
                for author in json.loads(str(metadata[_AUTHORS_KEY]))
            ),
            abstract=None if abstract is None else str(abstract),
            metadata={
                key: value
                for key, value in metadata.items()
                if not key.startswith(RESERVED_KEY_PREFIX)
            },
        )

    def _to_similarity(self, distance: float) -> float:
        """Convert a ChromaDB distance into a similarity where higher is better.

        Because every embedding is L2-normalized, each metric maps to cosine
        similarity:

        * ``cosine``: similarity = 1 - distance
        * ``ip``: similarity = 1 - distance
        * ``l2`` (ChromaDB reports squared L2): similarity = 1 - distance / 2
        """
        if self._distance_metric == "l2":
            return 1.0 - distance / 2.0
        return 1.0 - distance
