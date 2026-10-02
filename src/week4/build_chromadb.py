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

import itertools
import logging
import time
from dataclasses import dataclass

from src.week4.config import CONFIG, Week4Config
from src.week4.data_adapter import Week2PaperSource
from src.week4.embedder import Embedder
from src.week4.interfaces import PaperSource
from src.week4.vector_store import VectorStore

logger = logging.getLogger(__name__)

_PROGRESS_EVERY_BATCHES = 20


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
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
    start = time.perf_counter()
    papers_indexed = 0
    batch_number = 0
    for batch_number, batch in enumerate(
        itertools.batched(source.load_papers(), batch_size), start=1
    ):
        try:
            store.add_papers(batch, embedder.embed_documents([paper.text for paper in batch]))
        except Exception:
            logger.error(
                "Indexing failed at batch %d (first paper_id %s).", batch_number, batch[0].paper_id
            )
            raise
        papers_indexed += len(batch)
        logger.debug("Batch %d stored; %d papers indexed so far.", batch_number, papers_indexed)
        if batch_number % _PROGRESS_EVERY_BATCHES == 0:
            logger.info("%d papers indexed so far.", papers_indexed)
    if papers_indexed == 0:
        raise RuntimeError("The paper source yielded no papers, so nothing was indexed.")
    return IndexingStats(
        papers_indexed=papers_indexed,
        batches=batch_number,
        collection_size=store.count(),
        duration_seconds=time.perf_counter() - start,
    )


# TODO(future): move shared construction into a small src/week4/factories.py
# (build_embedder, build_vector_store, build_retriever, build_aggregator) so
# this module and week4_pipeline.build_week4_pipeline reuse the same wiring
# instead of duplicating it.
def _build_components(config: Week4Config) -> tuple[PaperSource, Embedder, VectorStore]:
    """Create the paper source, embedder and vector store from configuration.

    The store receives ``embedder.model_name`` so the index records which
    model built it.
    """
    source = Week2PaperSource(config.papers_path, config.authorship_path)
    embedder = Embedder(
        config.embedding_model_name, device=config.device, batch_size=config.batch_size
    )
    store = VectorStore(
        config.chroma_dir,
        config.collection_name,
        distance_metric=config.distance_metric,
        embedding_model_name=embedder.model_name,
    )
    return source, embedder, store


def main(config: Week4Config = CONFIG) -> None:
    """Build or update the ChromaDB index and report indexing statistics."""
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Third-party libraries stay at WARNING; Week 4 modules, including this one
    # when it runs as __main__, log at the configured level.
    for name in ("src.week4", __name__):
        logging.getLogger(name).setLevel(config.log_level)

    source, embedder, store = _build_components(config)
    logger.info(
        "Indexing with %s into collection %r at %s (batch size %d).",
        embedder.model_name,
        config.collection_name,
        config.chroma_dir,
        config.batch_size,
    )
    stats = build_index(source, embedder, store, batch_size=config.batch_size)
    logger.info(
        "Indexed %d papers in %d batches in %.1fs (%.0f papers/s); collection size %d.",
        stats.papers_indexed,
        stats.batches,
        stats.duration_seconds,
        stats.papers_indexed / stats.duration_seconds,
        stats.collection_size,
    )
    if stats.collection_size > stats.papers_indexed:
        logger.warning(
            "The collection still holds %d papers from earlier runs that are no longer in "
            "the source data; delete %s and rebuild to remove them.",
            stats.collection_size - stats.papers_indexed,
            config.chroma_dir,
        )


if __name__ == "__main__":
    main()
