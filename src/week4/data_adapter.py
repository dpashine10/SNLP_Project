"""Adapter from the Week 2 dataset to Week 4 objects.

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

Week 2 sources
--------------
``src/week2/buildprofiles.py`` writes the paper inputs:

* ``papers.csv``: ``work_id``, ``title``, ``paper_text``, plus ``abstract``
  and the scalar metadata columns when present.
* ``paper_author.csv``: ``work_id``, ``author_id``, ``author_name``.

``src/week2/create_evaluation_set.py`` writes the labelled benchmark read by
``load_labeled_queries``: ``query``, ``author_id``, ``relevant`` (0 or 1).
"""

import logging
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pandas as pd

from src.week4.schemas import Author, LabeledQuery, MetadataValue, Paper

logger = logging.getLogger(__name__)

RawRecord = Mapping[str, str]

_PAPER_COLUMNS = ("work_id", "title", "paper_text")
_METADATA_COLUMNS = ("publication_year", "doi", "institutions", "topics", "keywords")
_AUTHOR_COLUMNS = ("work_id", "author_id", "author_name")
_LABEL_COLUMNS = ("query", "author_id", "relevant")
_READ_CHUNK_ROWS = 5_000


class Week2PaperSource:
    """Reads Week 2 output and yields ``Paper`` objects.

    Satisfies the ``PaperSource`` protocol.

    Data-quality policy (decided here, never downstream):

    * Records that cannot produce a non-blank ``Paper.text`` are skipped,
      because the embedder rejects blank text.
    * Papers without authors are skipped: they cannot contribute to
      researcher ranking and would only take retrieval slots.
    * Duplicate paper IDs keep the first occurrence, so ``paper_id`` values
      are unique as the protocol requires.
    * Every skipped or deduplicated record is counted and reported in a
      warning summary; nothing is dropped silently.

    When loading finishes, a summary logs the records read, papers yielded,
    papers skipped (by reason) and papers kept without an abstract.
    """

    def __init__(self, papers_path: Path | None, authorship_path: Path | None) -> None:
        """Create the adapter without reading any files.

        Args:
            papers_path: Week 2 paper-level output, normally ``config.papers_path``.
            authorship_path: Week 2 paper-author relationships, normally
                ``config.authorship_path``.

        Raises:
            ValueError: If either path is ``None``, with a message pointing
                to ``config.py``.
        """
        if papers_path is None or authorship_path is None:
            raise ValueError(
                "Week 2 input paths are not set: configure papers_path and "
                "authorship_path in src/week4/config.py."
            )
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
        authors_by_paper = self._load_authors()
        counts: Counter[str] = Counter()
        seen_ids: set[str] = set()
        for record in self._read_paper_records():
            counts["records read"] += 1
            paper_id = record["work_id"].strip()
            text = self._build_text(record)
            authors = authors_by_paper.get(paper_id, ())
            if not paper_id or not text:
                counts["skipped (no ID or text)"] += 1
            elif paper_id in seen_ids:
                counts["skipped (duplicate ID)"] += 1
            elif not authors:
                counts["skipped (no authors)"] += 1
            else:
                seen_ids.add(paper_id)
                if not record.get("abstract", "").strip():
                    counts["kept without abstract"] += 1
                counts["papers yielded"] += 1
                yield self._to_paper(record, text, authors)
        self._log_summary(counts)

    def _read_paper_records(self) -> Iterator[RawRecord]:
        """Stream raw paper records from the Week 2 papers file.

        Every value is read as a string, with blanks as ``""``.

        Raises:
            FileNotFoundError: If the papers file does not exist.
            ValueError: If required columns are missing.
        """
        header = _read_header(self._papers_path, _PAPER_COLUMNS)
        optional = [column for column in ("abstract", *_METADATA_COLUMNS) if column in header]
        chunks = pd.read_csv(
            self._papers_path,
            usecols=[*_PAPER_COLUMNS, *optional],
            dtype=str,
            keep_default_na=False,
            chunksize=_READ_CHUNK_ROWS,
        )
        for chunk in chunks:
            yield from chunk.to_dict("records")

    def _load_authors(self) -> dict[str, tuple[Author, ...]]:
        """Map each paper ID to its authors.

        Returns:
            A mapping from Week 2 paper ID to that paper's authors, in source
            order, with each author listed once per paper.

        Raises:
            FileNotFoundError: If the authorship file does not exist.
            ValueError: If required columns are missing.
        """
        _read_header(self._authorship_path, _AUTHOR_COLUMNS)
        links = pd.read_csv(
            self._authorship_path,
            usecols=list(_AUTHOR_COLUMNS),
            dtype=str,
            keep_default_na=False,
        )
        authors: dict[str, list[Author]] = {}
        seen_links: set[tuple[str, str]] = set()
        for work_id, author_id, author_name in links[list(_AUTHOR_COLUMNS)].itertuples(
            index=False, name=None
        ):
            link = (work_id.strip(), author_id.strip())
            if not all(link) or link in seen_links:
                continue
            seen_links.add(link)
            authors.setdefault(link[0], []).append(
                Author(author_id=link[1], name=author_name.strip())
            )
        return {work_id: tuple(paper_authors) for work_id, paper_authors in authors.items()}

    @staticmethod
    def _build_text(record: RawRecord) -> str:
        """Return the text to embed for one paper.

        Reuses Week 2's ``paper_text`` (title, abstract, topics and keywords,
        falling back to the title) instead of composing a separate text.

        Returns:
            The text to embed, or an empty string if the record has none
            (the caller then skips it).
        """
        return record["paper_text"].strip()

    @staticmethod
    def _to_paper(record: RawRecord, text: str, authors: tuple[Author, ...]) -> Paper:
        """Convert one usable raw record into a ``Paper``."""
        abstract = record.get("abstract", "").strip()
        return Paper(
            paper_id=record["work_id"].strip(),
            title=record["title"].strip(),
            text=text,
            authors=authors,
            abstract=abstract or None,
            metadata=_scalar_metadata(record),
        )

    @staticmethod
    def _log_summary(counts: Counter[str]) -> None:
        """Log the loading summary, as a warning if any record was skipped."""
        summary = ", ".join(f"{name}: {count:,}" for name, count in counts.items())
        if any(name.startswith("skipped") for name in counts):
            logger.warning("Week 2 papers loaded with skips (%s).", summary)
        else:
            logger.info("Week 2 papers loaded (%s).", summary)


def load_labeled_queries(labels_path: Path | None) -> list[LabeledQuery]:
    """Load Week 2's labelled benchmark as ``LabeledQuery`` objects.

    One query per distinct ``query`` value, in file order. A researcher is
    relevant to a query when its ``relevant`` value is 1.

    Args:
        labels_path: The labelled benchmark, normally
            ``config.evaluation_labels_path``.

    Returns:
        The labelled queries, including any with no relevant researcher.

    Raises:
        ValueError: If ``labels_path`` is ``None``, a required column is
            missing or a ``relevant`` value is not 0 or 1.
        FileNotFoundError: If the label file does not exist.
    """
    if labels_path is None:
        raise ValueError(
            "evaluation_labels_path is not set: configure it in src/week4/config.py."
        )
    _read_header(labels_path, _LABEL_COLUMNS)
    labels = pd.read_csv(labels_path, usecols=list(_LABEL_COLUMNS), dtype=str, keep_default_na=False)
    relevant_by_query: dict[str, set[str]] = {}
    for query, author_id, relevant in labels[list(_LABEL_COLUMNS)].itertuples(
        index=False, name=None
    ):
        if relevant.strip() not in ("0", "1"):
            raise ValueError(
                f"{labels_path.name}: 'relevant' must be 0 or 1, got {relevant!r} "
                f"for query {query!r}."
            )
        relevant_ids = relevant_by_query.setdefault(query.strip(), set())
        if relevant.strip() == "1":
            relevant_ids.add(author_id.strip())
    return [
        LabeledQuery(query=query, relevant_author_ids=frozenset(relevant_ids))
        for query, relevant_ids in relevant_by_query.items()
    ]


def _read_header(path: Path, required: Sequence[str]) -> list[str]:
    """Return a Week 2 CSV's header after checking the file and required columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a required column is missing.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Week 2 file not found: {path}")
    header = list(pd.read_csv(path, nrows=0).columns)
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(
            f"{path} is missing Week 2 column(s): {', '.join(missing)}. If the file "
            "is a Git LFS pointer, run `git lfs pull` first."
        )
    return header


def _scalar_metadata(record: RawRecord) -> dict[str, MetadataValue]:
    """Collect the non-blank metadata columns, with the year as an integer when possible."""
    metadata: dict[str, MetadataValue] = {}
    for column in _METADATA_COLUMNS:
        value = record.get(column, "").strip()
        if not value:
            continue
        metadata[column] = int(value) if column == "publication_year" and value.isdigit() else value
    return metadata
