"""Internal data objects shared by Week 4 modules.

These dataclasses are the contract between modules. The adapters convert
teammate data into these objects and every other module consumes only these,
so a change in the Week 2 or Week 3 format never spreads past the adapters.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

MetadataValue = str | int | float | bool


@dataclass(frozen=True, slots=True)
class Author:
    """A researcher who authored at least one indexed paper.

    Attributes:
        author_id: Stable unique identifier, used to group papers by author
            and to match evaluation labels.
        name: Display name. Not guaranteed to be unique.
    """

    author_id: str
    name: str


# Extra attributes (institution, department, publication year, DOI, OpenAlex ID,
# etc.) are intentionally kept inside ``metadata`` rather than added as fields,
# so this schema stays stable even if the Week 2 output evolves.
@dataclass(frozen=True, slots=True)
class Paper:
    """A single paper to be indexed.

    Attributes:
        paper_id: Stable unique identifier, used as the ChromaDB document ID.
        title: Display title, shown as evidence for a recommendation.
        text: The exact text to embed. Built by ``data_adapter`` so the choice
            of source fields lives in one place.
        authors: Authors of the paper, in source order.
        abstract: Display abstract, if available. Kept separate from ``text``
            because embedded text and displayed text serve different purposes.
        metadata: Optional scalar attributes (e.g. publication year) stored
            alongside the embedding for filtering and display.
    """

    paper_id: str
    title: str
    text: str
    authors: tuple[Author, ...]
    abstract: str | None = None
    # Scalar values only (str, int, float, bool), matching what ChromaDB accepts
    # as metadata. Do not store lists, dicts or other nested Python objects here.
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict, hash=False)


@dataclass(frozen=True, slots=True)
class PaperHit:
    """A paper returned by vector search for a query.

    Attributes:
        paper: The matched paper.
        score: Similarity to the query. Higher score = better match,
            regardless of which distance metric the vector backend uses.
            The vector store converts backend distances into this
            convention, so no other module depends on ChromaDB details.
    """

    paper: Paper
    score: float


@dataclass(frozen=True, slots=True)
class ResearcherScore:
    """One researcher's result for a query.

    Sequences of ``ResearcherScore`` are ordered best-first; a researcher's
    rank is their position in that sequence.

    Attributes:
        author: The recommended researcher.
        score: Relevance produced by the active ranking strategy.
        evidence: The paper hits behind the score, best-first, used to explain
            the recommendation. Empty when a recommender (e.g. the baseline)
            does not expose paper-level evidence.
        matched_papers: Number of retrieved papers by this researcher that were
            passed to the ranking strategy. Informational only (evaluation and
            debugging); it never affects ranking.
    """

    author: Author
    score: float
    evidence: tuple[PaperHit, ...] = ()
    matched_papers: int = 0


@dataclass(frozen=True, slots=True)
class LabeledQuery:
    """A query with its ground-truth relevant researchers.

    Attributes:
        query: Query text sent to every recommender.
        relevant_author_ids: Canonical IDs of researchers judged relevant.
    """

    query: str
    relevant_author_ids: frozenset[str]
