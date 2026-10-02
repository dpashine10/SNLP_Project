"""Indexing entry point: builds the persistent ChromaDB index of papers.

Orchestration only::

    PaperSource.load_papers() -> batches of Papers
      -> Embedder.embed_documents() -> VectorStore.add_papers()

It never reads CSV files or knows their schema (``data_adapter`` does), never
computes embeddings itself (``Embedder`` does) and never calls ChromaDB APIs
(``VectorStore`` does). Papers are streamed, so only one batch of papers and
their embeddings is held in memory at a time.

Restartable: ``VectorStore`` writes are upserts keyed by ``paper_id``, so if
a run fails part-way, running it again completes the index without creating
duplicates.

Run from the repository root::

    python -m src.week4.build_chromadb
"""

# TODO(future): a --rebuild flag that deletes and recreates the collection
# before indexing (requires a VectorStore method to delete the collection).
# TODO(future): resume support that skips paper IDs already in the collection
# instead of re-embedding them.

import logging
from dataclasses import dataclass

from src.week4.config import CONFIG, Week4Config
from src.week4.embedder import Embedder
from src.week4.interfaces import PaperSource
from src.week4.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexingStats:
    """Summary of one indexing run.

    Attributes:
        papers_indexed: Papers embedded and upserted during this run.
        batches: Number of batches processed.
        collection_size: Papers in the collection after the run. Larger than
            ``papers_indexed`` when the collection still holds papers from
            earlier runs that are no longer in the source data.
        duration_seconds: Wall-clock duration of the run.
    """

    papers_indexed: int
    batches: int
    collection_size: int
    duration_seconds: float


def build_index(
    source: PaperSource,
    embedder: Embedder,
    store: VectorStore,
    *,
    batch_size: int,
) -> IndexingStats:
    """Stream papers from ``source``, embed them in batches and store them.

    Each batch is embedded and upserted before the next batch is read. If a
    batch fails, the batch number and its first ``paper_id`` are logged and
    the original exception is re-raised unchanged; re-running is safe
    because writes are upserts.

    Args:
        source: Supplier of papers, normally ``Week2PaperSource``.
        embedder: Embedder for paper text.
        store: Destination index, configured with ``embedder.model_name``.
        batch_size: Papers per batch.

    Returns:
        Statistics for this run.

    Raises:
        ValueError: If ``batch_size`` is less than 1, or propagated from the
            embedder or store (e.g. blank text, duplicate paper IDs).
        RuntimeError: If ``source`` yields no papers, or propagated from the
            embedder or store (e.g. model load failure, incompatible index).
        FileNotFoundError: Propagated from ``source`` if input data is missing.
    """
    # TODO(Phase 3): validate batch_size; iterate
    # itertools.batched(source.load_papers(), batch_size); embed
    # [paper.text for paper in batch]; store.add_papers(batch, embeddings);
    # count papers and batches; time the run with time.perf_counter; read
    # store.count() at the end. Log the current batch number and running
    # paper count at DEBUG level.
    raise NotImplementedError


# TODO(future): move shared construction into a small src/week4/factories.py
# (build_embedder, build_vector_store, build_retriever, build_aggregator) so
# this module and week4_pipeline.build_week4_pipeline reuse the same wiring
# instead of duplicating it. Not part of Phase 1.
def _build_components(config: Week4Config) -> tuple[PaperSource, Embedder, VectorStore]:
    """Create the paper source, embedder and vector store from configuration.

    The store receives ``embedder.model_name`` so the index records which
    model built it.
    """
    # TODO(Phase 3): construct Week2PaperSource(config.papers_path,
    # config.authorship_path), then Embedder and VectorStore from config.
    raise NotImplementedError


def main(config: Week4Config = CONFIG) -> None:
    """Build or update the ChromaDB index and report indexing statistics."""
    # TODO(Phase 3): configure logging at config.log_level; log the embedding
    # model, collection name, Chroma directory and batch size; run
    # build_index; log papers indexed, batches, indexing duration, indexing
    # throughput (papers/sec) and final collection size. If collection_size >
    # papers_indexed, warn that papers from earlier runs remain in the index
    # and suggest a rebuild.
    raise NotImplementedError


if __name__ == "__main__":
    main()
