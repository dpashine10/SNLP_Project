"""Adapter from the Week 2 dataset to Week 4 ``Paper`` objects.

This is the ONLY module that understands the Week 2 dataset: CSV schema,
column names, file layout, preprocessing outputs and data-cleaning rules all
live here. No other Week 4 module accesses raw Week 2 data. It never
generates embeddings, accesses ChromaDB, retrieves papers or ranks
researchers. When Week 2 changes, only this module (and the paths in
``config``) should change.

What Week 4 needs from Week 2, per paper
----------------------------------------
* a stable unique paper ID                  -> ``Paper.paper_id``
* a title                                   -> ``Paper.title``
* one or more text fields to embed          -> ``Paper.text``
* its authors (stable author ID + name)     -> ``Paper.authors``
* optionally an abstract                    -> ``Paper.abstract``
* optionally extra scalar attributes        -> ``Paper.metadata``
  (e.g. year, venue, DOI; keys must not start with ``__``)

Which Week 2 files and columns provide each of these is unknown until the
final Week 2 code is pulled; see the ``TODO(Phase 2)`` markers.
"""

import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from src.week4.schemas import Author, Paper

logger = logging.getLogger(__name__)

RawRecord = Mapping[str, Any]


class Week2PaperSource:
    """Reads Week 2 output and yields ``Paper`` objects.

    Satisfies the ``PaperSource`` protocol.

    Data-quality policy (decided here, never downstream):

    * Records that cannot produce a non-blank ``Paper.text`` are skipped,
      because the embedder rejects blank text.
    * Duplicate paper IDs keep the first occurrence, so ``paper_id`` values
      are unique as the protocol requires.
    * Every skipped or deduplicated record is counted and reported in a
      warning summary; nothing is dropped silently.

    Logging (Phase 2/3): total records read, valid papers, skipped papers,
    duplicate papers, papers missing authors and papers missing abstracts.
    """

    # TODO(Phase 2): decide the policy for papers without authors after
    # inspecting the final Week 2 data:
    #   Option A: skip them, because they cannot contribute to researcher ranking.
    #   Option B: index them anyway, to support future paper-only search.

    def __init__(self, papers_path: Path | None, authorship_path: Path | None = None) -> None:
        """Create the adapter without reading any files.

        Args:
            papers_path: Week 2 paper-level output, normally
                ``config.papers_path``.
            authorship_path: Week 2 paper-author relationships, if stored in
                a separate file; normally ``config.authorship_path``.

        Raises:
            ValueError: If ``papers_path`` is ``None``, with a message pointing
                to ``papers_path`` in ``config.py``.
        """
        # TODO(Phase 3): raise the actionable ValueError when papers_path is None.
        self._papers_path = papers_path
        self._authorship_path = authorship_path

    def load_papers(self) -> Iterator[Paper]:
        """Yield every usable Week 2 paper as a ``Paper``.

        Papers are yielded lazily so large datasets can be indexed in batches.
        Author information is loaded first, then paper records are streamed.

        Returns:
            An iterator of ``Paper`` objects with unique ``paper_id`` values.

        Raises:
            FileNotFoundError: If a configured input file does not exist.
            ValueError: If required Week 2 columns are missing.
        """
        # TODO(Phase 2): once the Week 2 schema is known, implement:
        #   authors = _load_authors(); for each record in _read_paper_records():
        #   build with _to_paper, skip/deduplicate per the policy, yield.
        #   Log the summary counts listed in the class docstring.
        raise NotImplementedError

    def _read_paper_records(self) -> Iterator[RawRecord]:
        """Stream raw paper records from the Week 2 papers file.

        Raises:
            FileNotFoundError: If the papers file does not exist.
            ValueError: If required columns are missing.
        """
        # TODO(Phase 2): read papers_path (file format, reader library and
        # chunking depend on the final Week 2 output) and check that the
        # required columns exist.
        raise NotImplementedError

    def _load_authors(self) -> Mapping[str, tuple[Author, ...]]:
        """Map each paper ID to its authors.

        Returns:
            A mapping from Week 2 paper ID to that paper's authors, in source order.

        Raises:
            FileNotFoundError: If the authorship file does not exist.
            ValueError: If required columns are missing.
        """
        # TODO(Phase 2): authors may live in the papers file or in a separate
        # file at authorship_path; implement whichever Week 2 provides.
        raise NotImplementedError

    @staticmethod
    def _build_text(record: RawRecord) -> str:
        """Compose the text to embed for one paper.

        Returns:
            The text to embed, or an empty string if the record has no
            usable text (the caller then skips it).
        """
        # TODO(Phase 2): decide which Week 2 fields form the embedded text
        # (e.g. title, abstract, topics, keywords) and the fallback when the
        # abstract is missing.
        raise NotImplementedError

    def _to_paper(
        self, record: RawRecord, authors: Mapping[str, tuple[Author, ...]]
    ) -> Paper | None:
        """Convert one raw record into a ``Paper``.

        Returns:
            The ``Paper``, or ``None`` if the record must be skipped (e.g. no
            usable text or missing ID), so the caller can count it.
        """
        # TODO(Phase 2): map Week 2 columns to Paper fields, including
        # optional abstract and metadata. Forward only scalar values
        # (str, int, float, bool) into Paper.metadata; never lists or nested
        # objects, because VectorStore stores metadata in ChromaDB.
        raise NotImplementedError
