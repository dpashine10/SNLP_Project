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

        return cls(
            query=query,
            top_k=top_k,
            strategy=strategy,
            min_confidence=float(min_confidence),
            include_evidence=include_evidence,
            paper_top_k=paper_top_k,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperEvidence:
    """Individual paper hit serving as explainable evidence for a recommendation."""

    paper_id: str
    title: str
    year: str | int
    similarity: float
    authors_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "year": str(self.year),
            "similarity": round(float(self.similarity), 4),
            "authors_count": int(self.authors_count),
        }


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "execution_time_ms": round(float(self.execution_time_ms), 2),
            "total_results": self.total_results,
            "model_info": self.model_info.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)
