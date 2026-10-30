"""Structured input and output schemas for the Week 6 Reusable NLP Asset.

All schemas are immutable, typed dataclasses with built-in validation,
safe serialization/deserialization, and clear error reporting.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal

SUPPORTED_STRATEGIES = (
    "top_k_average_author_discount",
    "top_k_average",
    "best_paper",
    "average",
)
DEFAULT_STRATEGY = "top_k_average_author_discount"
MIN_TOP_K = 1
MAX_TOP_K = 50
DEFAULT_TOP_K = 5
DEFAULT_PAPER_TOP_K = 100
MAX_QUERY_LENGTH = 1000
MAX_PAPER_TOP_K = 200
MAX_BATCH_SIZE = 50
MAX_CLASSIFY_PAPERS = 50

PAPER_TYPES = (
    "empirical",
    "theoretical",
    "survey",
    "resource",
    "systems",
)
PAPER_TYPE_LABELS: dict[str, str] = {
    "empirical": "Empirical Study",
    "theoretical": "Theoretical",
    "survey": "Survey / Review",
    "resource": "Dataset / Benchmark",
    "systems": "Applied Systems",
}
RELEVANCE_GRADE_LABELS: dict[int, str] = {
    0: "Off-Topic (0/3)",
    1: "Tangential (1/3)",
    2: "Related Sub-Field (2/3)",
    3: "Direct Match (3/3)",
}


class ConfidenceLevel(str, Enum):
    """Calibrated confidence rating bands."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ValidationError(ValueError):
    """Raised when an incoming request fails validation."""

    def __init__(self, message: str, field_name: str | None = None) -> None:
        super().__init__(message)
        self.field_name = field_name
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": "Validation Error",
            "field": self.field_name,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class RecommendRequest:
    """Validated input contract for a researcher recommendation query.

    Attributes:
        query: Non-empty natural language research description or synopsis.
        top_k: Maximum number of researchers to return (1-50, default 5).
        strategy: Ranking strategy to combine paper hits into an author score.
        min_confidence: Minimum calibrated confidence (0.0 to 1.0) to filter results.
        include_evidence: Whether to return paper-level evidence hits.
        paper_top_k: Maximum initial paper hits to retrieve before aggregation (1-200).
    """

    query: str
    top_k: int = DEFAULT_TOP_K
    strategy: str = DEFAULT_STRATEGY
    min_confidence: float = 0.0
    include_evidence: bool = True
    paper_top_k: int = DEFAULT_PAPER_TOP_K
    classify_papers: bool = False
    include_reasoning: bool = False
    reasoning_provider: str = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValidationError("Query must be a non-empty string.", field_name="query")
        object.__setattr__(self, "query", self.query.strip())
        if len(self.query) > MAX_QUERY_LENGTH:
            raise ValidationError(
                f"Query exceeds maximum allowed length of {MAX_QUERY_LENGTH} characters.",
                field_name="query",
            )
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or not (MIN_TOP_K <= self.top_k <= MAX_TOP_K):
            raise ValidationError(
                f"top_k must be an integer between {MIN_TOP_K} and {MAX_TOP_K}.",
                field_name="top_k",
            )
        if not isinstance(self.strategy, str) or self.strategy not in SUPPORTED_STRATEGIES:
            supported = ", ".join(SUPPORTED_STRATEGIES)
            raise ValidationError(
                f"Unknown strategy {self.strategy!r}. Supported strategies: {supported}.",
                field_name="strategy",
            )
        if (
            isinstance(self.min_confidence, bool)
            or not isinstance(self.min_confidence, (int, float))
            or not (0.0 <= self.min_confidence <= 1.0)
        ):
            raise ValidationError(
                "min_confidence must be a float between 0.0 and 1.0.",
                field_name="min_confidence",
            )
        if not isinstance(self.include_evidence, bool):
            raise ValidationError("include_evidence must be a boolean.", field_name="include_evidence")
        if (
            isinstance(self.paper_top_k, bool)
            or not isinstance(self.paper_top_k, int)
            or not (1 <= self.paper_top_k <= MAX_PAPER_TOP_K)
        ):
            raise ValidationError(
                f"paper_top_k must be an integer between 1 and {MAX_PAPER_TOP_K}.",
                field_name="paper_top_k",
            )
        if not isinstance(self.classify_papers, bool):
            raise ValidationError("classify_papers must be a boolean.", field_name="classify_papers")
        if not isinstance(self.include_reasoning, bool):
            raise ValidationError("include_reasoning must be a boolean.", field_name="include_reasoning")
        if (
            not isinstance(self.reasoning_provider, str)
            or self.reasoning_provider.lower().strip() not in {"auto", "groq", "gemini"}
        ):
            raise ValidationError(
                "reasoning_provider must be 'auto', 'groq', or 'gemini'.",
                field_name="reasoning_provider",
            )
        object.__setattr__(self, "reasoning_provider", self.reasoning_provider.lower().strip())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RecommendRequest:
        """Create a validated RecommendRequest from a dictionary (e.g. JSON payload)."""
        if not isinstance(data, Mapping):
            raise ValidationError("Request payload must be a JSON object.")

        query = data.get("query", "")
        if not isinstance(query, str):
            raise ValidationError("Query must be a non-empty string.", field_name="query")

        top_k = data.get("top_k", DEFAULT_TOP_K)
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValidationError("top_k must be an integer.", field_name="top_k")

        strategy = data.get("strategy", DEFAULT_STRATEGY)
        if not isinstance(strategy, str):
            raise ValidationError("strategy must be a string.", field_name="strategy")

        min_confidence = data.get("min_confidence", 0.0)
        if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
            raise ValidationError("min_confidence must be a number.", field_name="min_confidence")

        include_evidence = data.get("include_evidence", True)
        if not isinstance(include_evidence, bool):
            raise ValidationError("include_evidence must be a boolean.", field_name="include_evidence")

        paper_top_k = data.get("paper_top_k", DEFAULT_PAPER_TOP_K)
        if isinstance(paper_top_k, bool) or not isinstance(paper_top_k, int):
            raise ValidationError("paper_top_k must be an integer.", field_name="paper_top_k")

        classify_papers = data.get("classify_papers", False)
        if not isinstance(classify_papers, bool):
            raise ValidationError("classify_papers must be a boolean.", field_name="classify_papers")

        include_reasoning = data.get("include_reasoning", False)
        if not isinstance(include_reasoning, bool):
            raise ValidationError("include_reasoning must be a boolean.", field_name="include_reasoning")

        reasoning_provider = str(data.get("reasoning_provider", "auto"))

        return cls(
            query=query,
            top_k=top_k,
            strategy=strategy,
            min_confidence=float(min_confidence),
            include_evidence=include_evidence,
            paper_top_k=paper_top_k,
            classify_papers=classify_papers,
            include_reasoning=include_reasoning,
            reasoning_provider=reasoning_provider,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperClassification:
    """Laya System-1 structured decision output for a research publication."""

    paper_id: str
    title: str
    paper_type: str
    paper_type_label: str
    type_confidence: float
    type_probabilities: dict[str, float]
    is_scholarly: bool
    scholarly_probability: float
    relevance_score: float | None = None
    relevance_grade: int | None = None
    relevance_label: str = ""
    abstained: bool = False
    backend: str = "laya-typed-decisions"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "paper_id": self.paper_id,
            "title": self.title,
            "paper_type": self.paper_type,
            "paper_type_label": self.paper_type_label,
            "type_confidence": round(float(self.type_confidence), 4),
            "type_probabilities": {k: round(float(v), 4) for k, v in self.type_probabilities.items()},
            "is_scholarly": bool(self.is_scholarly),
            "scholarly_probability": round(float(self.scholarly_probability), 4),
            "abstained": bool(self.abstained),
            "backend": self.backend,
        }
        if self.relevance_score is not None:
            payload["relevance_score"] = round(float(self.relevance_score), 4)
            payload["relevance_grade"] = int(self.relevance_grade) if self.relevance_grade is not None else 0
            payload["relevance_label"] = self.relevance_label
        return payload


@dataclass(frozen=True, slots=True)
class QueryClassification:
    """Laya System-1 intent and domain classification for a search query."""

    query: str
    primary_domain: str
    primary_domain_label: str
    domain_confidence: float
    domain_probabilities: dict[str, float]
    is_analytical_question: bool
    analytical_probability: float
    backend: str = "laya-typed-decisions"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "primary_domain": self.primary_domain,
            "primary_domain_label": self.primary_domain_label,
            "domain_confidence": round(float(self.domain_confidence), 4),
            "domain_probabilities": {k: round(float(v), 4) for k, v in self.domain_probabilities.items()},
            "is_analytical_question": bool(self.is_analytical_question),
            "analytical_probability": round(float(self.analytical_probability), 4),
            "backend": self.backend,
        }


@dataclass(frozen=True, slots=True)
class PaperEvidence:
    """Individual paper hit serving as explainable evidence for a recommendation."""

    paper_id: str
    title: str
    year: str | int
    similarity: float
    authors_count: int
    classification: PaperClassification | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "paper_id": self.paper_id,
            "title": self.title,
            "year": str(self.year),
            "similarity": round(float(self.similarity), 4),
            "authors_count": int(self.authors_count),
        }
        if self.classification is not None:
            out["classification"] = self.classification.to_dict()
        return out


@dataclass(frozen=True, slots=True)
class ConfidenceScore:
    """Calibrated confidence metric for a recommendation result."""

    score: float
    level: str
    raw_score: float
    margin: float
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(float(self.score), 4),
            "level": self.level,
            "raw_score": round(float(self.raw_score), 4),
            "margin": round(float(self.margin), 4),
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class ResearcherRecommendation:
    """Ranked researcher recommendation output."""

    rank: int
    author_id: str
    name: str
    score: float
    confidence: ConfidenceScore
    matched_papers: int
    evidence: list[PaperEvidence] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "author_id": self.author_id,
            "name": self.name,
            "score": round(float(self.score), 4),
            "confidence": self.confidence.to_dict(),
            "matched_papers": self.matched_papers,
            "evidence": [e.to_dict() for e in self.evidence],
            "topics": list(self.topics),
        }


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Metadata regarding the underlying NLP models and configuration."""

    name: str
    strategy: str
    backend: str
    embedding_dim: int = 384
    version: str = "1.0.0"
    data_source: str = "OpenAlex publication metadata"
    confidence_note: str = "Confidence bands support ranking review; they are not calibrated probabilities."
    advisory_only: bool = True
    ranking_explanation: str = "Scores rank publication similarity; they do not verify teaching status, faculty role, recency, or mentor availability."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecommendResponse:
    """Structured output contract for the recommendation service."""

    query: str
    execution_time_ms: float
    total_results: int
    model_info: ModelInfo
    results: list[ResearcherRecommendation]
    query_classification: QueryClassification | None = None
    reasoning: ReasoningResponse | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": self.query,
            "execution_time_ms": round(float(self.execution_time_ms), 2),
            "total_results": self.total_results,
            "model_info": self.model_info.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }
        if self.query_classification is not None:
            payload["query_classification"] = self.query_classification.to_dict()
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning.to_dict()
        return payload

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True, slots=True)
class ClassifyPaperInput:
    """Validated single-paper payload for standalone Laya classification."""

    title: str
    abstract: str = ""
    topics: str = ""
    paper_id: str = ""
    similarity: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValidationError("Each paper must include a non-empty 'title' string.", field_name="title")
        object.__setattr__(self, "title", self.title.strip())
        if not isinstance(self.abstract, str):
            raise ValidationError("'abstract' must be a string.", field_name="abstract")
        if not isinstance(self.topics, str):
            raise ValidationError("'topics' must be a string.", field_name="topics")
        if not isinstance(self.paper_id, str):
            raise ValidationError("'paper_id' must be a string.", field_name="paper_id")
        if self.similarity is not None and (isinstance(self.similarity, bool) or not isinstance(self.similarity, (int, float))):
            raise ValidationError("'similarity' must be a number when provided.", field_name="similarity")

    @classmethod
    def from_dict(cls, data: Any, index: int = 0) -> ClassifyPaperInput:
        if isinstance(data, ClassifyPaperInput):
            return data
        if isinstance(data, str):
            return cls(title=data, paper_id=f"P{index + 1}")
        if not isinstance(data, Mapping):
            raise ValidationError(f"Paper at index {index} must be a JSON object or title string.", field_name="papers")
        sim = data.get("similarity")
        return cls(
            title=data.get("title", ""),
            abstract=data.get("abstract", "") or "",
            topics=data.get("topics", "") or "",
            paper_id=str(data.get("paper_id", data.get("work_id", f"P{index + 1}")) or f"P{index + 1}"),
            similarity=float(sim) if isinstance(sim, (int, float)) and not isinstance(sim, bool) else None,
        )


@dataclass(frozen=True, slots=True)
class ClassifyRequest:
    """Validated request contract for standalone Laya paper & query classification."""

    papers: tuple[ClassifyPaperInput, ...] = ()
    query: str = ""
    min_confidence: float = 0.35

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise ValidationError("'query' must be a string.", field_name="query")
        object.__setattr__(self, "query", self.query.strip())
        if len(self.query) > MAX_QUERY_LENGTH:
            raise ValidationError(
                f"Query exceeds maximum allowed length of {MAX_QUERY_LENGTH} characters.",
                field_name="query",
            )
        if not self.papers and not self.query:
            raise ValidationError(
                "Provide at least one item in 'papers' or a non-empty 'query' to classify.",
                field_name="papers",
            )
        if len(self.papers) > MAX_CLASSIFY_PAPERS:
            raise ValidationError(
                f"At most {MAX_CLASSIFY_PAPERS} papers may be classified per request.",
                field_name="papers",
            )
        if (
            isinstance(self.min_confidence, bool)
            or not isinstance(self.min_confidence, (int, float))
            or not (0.0 <= self.min_confidence <= 1.0)
        ):
            raise ValidationError("min_confidence must be a float between 0.0 and 1.0.", field_name="min_confidence")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ClassifyRequest:
        if not isinstance(data, Mapping):
            raise ValidationError("Request payload must be a JSON object.")
        raw_papers = data.get("papers", [])
        if raw_papers is None:
            raw_papers = []
        if not isinstance(raw_papers, list):
            raise ValidationError("'papers' must be a list of paper objects.", field_name="papers")
        parsed_papers = tuple(ClassifyPaperInput.from_dict(item, idx) for idx, item in enumerate(raw_papers))
        query = data.get("query", "")
        if query is None:
            query = ""
        if not isinstance(query, str):
            raise ValidationError("'query' must be a string.", field_name="query")
        min_conf = data.get("min_confidence")
        if min_conf is None:
            min_conf = 0.35
        if isinstance(min_conf, bool) or not isinstance(min_conf, (int, float)):
            raise ValidationError("min_confidence must be a number.", field_name="min_confidence")
        return cls(papers=parsed_papers, query=query, min_confidence=float(min_conf))


@dataclass(frozen=True, slots=True)
class ClassifyResponse:
    """Structured output contract for standalone Laya classification."""

    execution_time_ms: float
    backend: str
    papers: list[PaperClassification]
    query_classification: QueryClassification | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "execution_time_ms": round(float(self.execution_time_ms), 2),
            "backend": self.backend,
            "total_papers": len(self.papers),
            "papers": [p.to_dict() for p in self.papers],
        }
        if self.query_classification is not None:
            payload["query_classification"] = self.query_classification.to_dict()
        return payload

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)


REASONING_PROVIDERS: tuple[str, ...] = ("auto", "groq", "gemini")


@dataclass(frozen=True, slots=True)
class CitationSource:
    """Source publication providing grounding evidence for AI reasoning."""

    paper_index: int
    title: str
    paper_id: str = ""
    year: str = "N/A"
    abstract: str = ""
    doi: str | None = None
    authors: tuple[str, ...] = ()
    similarity: float = 0.0

    @classmethod
    def from_dict(cls, data: Any, index: int = 1) -> CitationSource:
        if isinstance(data, CitationSource):
            return data
        if isinstance(data, str):
            return cls(paper_index=index, title=data, paper_id=f"P{index}")
        if not isinstance(data, Mapping):
            raise ValidationError(
                f"Citation source at index {index} must be a JSON object or string.",
                field_name="papers",
            )
        raw_authors = data.get("authors") or []
        if isinstance(raw_authors, str):
            authors = tuple(a.strip() for a in raw_authors.replace("|", ",").split(",") if a.strip())
        elif isinstance(raw_authors, (list, tuple)):
            authors = tuple(str(a) for a in raw_authors)
        else:
            authors = ()
        return cls(
            paper_index=int(data.get("paper_index", index)),
            title=str(data.get("title", "")).strip() or "Untitled Publication",
            paper_id=str(data.get("paper_id", data.get("work_id", f"P{index}"))),
            year=str(data.get("year", "N/A")),
            abstract=str(data.get("abstract", "") or ""),
            doi=str(data.get("doi")) if data.get("doi") else None,
            authors=authors,
            similarity=float(data.get("similarity", data.get("paper_score", data.get("score", 0.0)))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_index": self.paper_index,
            "title": self.title,
            "paper_id": self.paper_id,
            "year": self.year,
            "abstract": self.abstract,
            "doi": self.doi,
            "authors": list(self.authors),
            "similarity": round(float(self.similarity), 4),
        }


@dataclass(frozen=True, slots=True)
class ReasoningRequest:
    """Validated input contract for dataset-grounded reasoning."""

    query: str
    papers: tuple[CitationSource, ...] = ()
    provider: str = "auto"
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValidationError("Query must be a non-empty string.", field_name="query")
        object.__setattr__(self, "query", self.query.strip())
        if not isinstance(self.provider, str) or self.provider.lower().strip() not in REASONING_PROVIDERS:
            raise ValidationError(
                f"provider must be one of {REASONING_PROVIDERS}.",
                field_name="provider",
            )
        object.__setattr__(self, "provider", self.provider.lower().strip())
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or not (64 <= self.max_tokens <= 4096)
        ):
            raise ValidationError(
                "max_tokens must be an integer between 64 and 4096.",
                field_name="max_tokens",
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReasoningRequest:
        if not isinstance(data, Mapping):
            raise ValidationError("Request payload must be a JSON object.")
        query = data.get("query", "")
        if not isinstance(query, str) or not query.strip():
            raise ValidationError("Query must be a non-empty string.", field_name="query")
        raw_papers = data.get("papers", []) or []
        if not isinstance(raw_papers, (list, tuple)):
            raise ValidationError("'papers' must be a list of citation source objects.", field_name="papers")
        parsed_papers = tuple(CitationSource.from_dict(p, idx + 1) for idx, p in enumerate(raw_papers))
        provider = str(data.get("provider", "auto"))
        max_tokens = data.get("max_tokens", 1024)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValidationError("max_tokens must be an integer.", field_name="max_tokens")
        return cls(query=query, papers=parsed_papers, provider=provider, max_tokens=max_tokens)


@dataclass(frozen=True, slots=True)
class ReasoningResponse:
    """Structured output contract for dataset-grounded reasoning."""

    query: str
    answer: str
    provider_used: str
    model_name: str
    citations: list[CitationSource]
    execution_time_ms: float
    is_fallback: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "provider_used": self.provider_used,
            "model_name": self.model_name,
            "citations": [c.to_dict() for c in self.citations],
            "execution_time_ms": round(float(self.execution_time_ms), 2),
            "is_fallback": self.is_fallback,
            "error": self.error,
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)
