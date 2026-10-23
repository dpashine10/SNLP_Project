"""Core Reusable NLP Asset for researcher recommendation.

Implements the unified Week 6 component with support for:
- Week 4/5 paper-level retrieval and aggregation
- The improved Week 5 `top_k_average_author_discount` ranking strategy
- Direct ChromaDB index or high-speed in-memory vector store
- Calibrated confidence scoring and explainable paper evidence
- Strict input/output schema contracts
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.week4.aggregator import Aggregator
from src.week4.config import DistanceMetric, RankingStrategyName, Week4Config
from src.week4.embedder import Embedder
from src.week4.ranking_strategy import create_strategy
from src.week4.schemas import Author, Paper, PaperHit, ResearcherScore
from src.week6.confidence import ConfidenceCalibrator
from src.week6.schemas import (
    DEFAULT_PAPER_TOP_K,
    DEFAULT_STRATEGY,
    DEFAULT_TOP_K,
    ModelInfo,
    PaperEvidence,
    RecommendRequest,
    RecommendResponse,
    ResearcherRecommendation,
    ValidationError,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class AssetConfig:
    """Configuration for the reusable NLP asset."""

    model_name: str = "all-MiniLM-L6-v2"
    default_strategy: str = DEFAULT_STRATEGY
    paper_top_k: int = DEFAULT_PAPER_TOP_K
    default_top_k: int = DEFAULT_TOP_K
    device: str = "auto"
    batch_size: int = 32
    chroma_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "Data" / "week4" / "chroma")
    # Keep the default aligned with src.week4.config so a built Chroma index
    # can be reused by the Week 6 service without a hidden collection mismatch.
    collection_name: str = "iitb_research_papers"
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "Data" / "processed")
    prefer_chroma: bool = True
    eager_load: bool = False


class _FastPaperVectorStore:
    """High-speed in-memory vector store using precomputed embeddings.

    Provides paper-level vector retrieval with sub-10ms query latency,
    identical in contract to ChromaDB VectorStore.
    """

    def __init__(
        self,
        npz_path: Path,
        papers_path: Path,
        authorship_path: Path,
        embedding_model_name: str,
    ) -> None:
        self.npz_path = npz_path
        self.papers_path = papers_path
        self.authorship_path = authorship_path
        self.model_name = embedding_model_name
        self._work_ids: np.ndarray | None = None
        self._embeddings: np.ndarray | None = None
        self._paper_lookup: dict[str, Paper] = {}
        self._loaded = False
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            logger.info("Loading precomputed paper vectors from %s...", self.npz_path)
            npz = np.load(self.npz_path)
            self._work_ids = np.asarray(npz["work_ids"], dtype=str)
            # Ensure float32 normalized embeddings
            emb = np.asarray(npz["embeddings"], dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._embeddings = emb / norms

            logger.info("Loading paper metadata from %s...", self.papers_path)
            df = pd.read_csv(
                self.papers_path,
                usecols=["work_id", "title", "publication_year", "authors", "author_ids", "topics", "keywords"],
                dtype=str,
                keep_default_na=False,
            )
            for row in df.itertuples(index=False):
                work_id = str(row.work_id)
                raw_authors = str(row.authors)
                raw_ids = str(row.author_ids)
                if not raw_authors or not raw_ids:
                    continue
                names = [a.strip() for a in raw_authors.split(";") if a.strip()]
                aids = [aid.strip() for aid in raw_ids.split("|") if aid.strip()]
                authors_list = []
                for idx, name in enumerate(names):
                    aid = aids[idx] if idx < len(aids) else f"auth_{name}"
                    authors_list.append(Author(author_id=aid, name=name))
                if not authors_list:
                    continue
                year = str(row.publication_year) if str(row.publication_year) else "N/A"
                topics = str(row.topics).strip()
                keywords = str(row.keywords).strip()
                self._paper_lookup[work_id] = Paper(
                    paper_id=work_id,
                    title=str(row.title),
                    text=str(row.title),
                    authors=tuple(authors_list),
                    metadata={"publication_year": year, "topics": topics, "keywords": keywords},
                )
            self._loaded = True
            logger.info("Fast paper vector store ready: %d papers indexed.", len(self._paper_lookup))

    def query(self, query_vector: np.ndarray, top_k: int) -> list[PaperHit]:
        if not self._loaded:
            self.load()
        assert self._embeddings is not None and self._work_ids is not None

        # Normalize query vector
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        # Dot-product cosine similarity
        similarities = np.dot(self._embeddings, query_vector)
        n = len(similarities)
        k = min(top_k * 2, n)  # Retrieve extra candidates to account for authorless papers
        top_indices = np.argpartition(similarities, -k)[-k:]
        sorted_indices = top_indices[np.argsort(-similarities[top_indices])]

        hits: list[PaperHit] = []
        for idx in sorted_indices:
            work_id = str(self._work_ids[idx])
            paper = self._paper_lookup.get(work_id)
            if paper is not None:
                score = float(similarities[idx])
                hits.append(PaperHit(paper=paper, score=score))
                if len(hits) >= top_k:
                    break
        return hits


class NLPAsset:
    """Reusable NLP Asset providing researcher recommendation services."""

    def __init__(self, config: AssetConfig | None = None) -> None:
        self.config = config or AssetConfig()
        self._embedder: Embedder | None = None
        self._vector_store: Any = None
        self._backend_type = "unknown"
        self._lock = threading.Lock()
        self._ready_event = threading.Event()
        self._load_error: str | None = None

        if self.config.eager_load:
            self.warmup()

    @property
    def is_ready(self) -> bool:
        return self._ready_event.is_set() and self._load_error is None

    def warmup(self) -> None:
        """Load models and index into memory."""
        with self._lock:
            if self._ready_event.is_set():
                return
            try:
                logger.info("Initializing Embedder with model %s...", self.config.model_name)
                self._embedder = Embedder(
                    self.config.model_name,
                    device=self.config.device,
                    batch_size=self.config.batch_size,
                )

                # Determine backend: ChromaDB or Fast NPZ
                chroma_path = Path(self.config.chroma_dir)
                has_chroma = (
                    self.config.prefer_chroma
                    and chroma_path.exists()
                    and any(chroma_path.glob("*.sqlite3"))
                )

                if has_chroma:
                    from src.week4.vector_store import VectorStore

                    logger.info("Using ChromaDB persistent store at %s", chroma_path)
                    collection_names = [self.config.collection_name]
                    # Older Week 6 runs used this name before the service was
                    # aligned with src.week4.config. Keep those local indexes
                    # readable instead of forcing a destructive rebuild.
                    if self.config.collection_name != "week4_papers":
                        collection_names.append("week4_papers")

                    store = None
                    last_error: RuntimeError | None = None
                    for collection_name in collection_names:
                        candidate = VectorStore(
                            chroma_path,
                            collection_name,
                            distance_metric=DistanceMetric.COSINE,
                            embedding_model_name=self._embedder.model_name,
                        )
                        try:
                            candidate._get_collection(create=False)
                        except RuntimeError as exc:
                            last_error = exc
                            continue
                        store = candidate
                        if collection_name != self.config.collection_name:
                            logger.warning(
                                "Using legacy ChromaDB collection %s; rebuild the index to use %s.",
                                collection_name,
                                self.config.collection_name,
                            )
                        break

                    if store is None:
                        assert last_error is not None
                        raise last_error
                    self._vector_store = store
                    self._backend_type = "chromadb_paper_level"
                else:
                    logger.info("Using high-performance in-memory paper vector store")
                    data_dir = Path(self.config.data_dir)
                    store = _FastPaperVectorStore(
                        npz_path=data_dir / "paper_embeddings.npz",
                        papers_path=data_dir / "papers.csv",
                        authorship_path=data_dir / "paper_author.csv",
                        embedding_model_name=self._embedder.model_name,
                    )
                    store.load()
                    self._vector_store = store
                    self._backend_type = "fast_npz_paper_level"

                self._ready_event.set()
                logger.info("NLP Asset warmed up successfully (backend=%s).", self._backend_type)
            except Exception as exc:
                self._load_error = str(exc)
                self._ready_event.set()
                logger.error("Failed to initialize NLP Asset: %s", exc, exc_info=True)
                raise

    def recommend(self, request: RecommendRequest | Mapping[str, Any] | str) -> RecommendResponse:
        """Recommend researchers matching a structured or text request.

        Args:
            request: A RecommendRequest, dictionary, or query string.

        Returns:
            Structured RecommendResponse containing ranked candidates,
            calibrated confidence scores, and supporting evidence papers.
        """
        # Coerce to validated RecommendRequest
        if isinstance(request, str):
            req = RecommendRequest(query=request)
        elif isinstance(request, Mapping):
            req = RecommendRequest.from_dict(request)
        elif isinstance(request, RecommendRequest):
            req = request
        else:
            raise ValidationError("Request must be a RecommendRequest, dict, or string.")

        if not self._ready_event.is_set():
            self.warmup()
        if self._load_error:
            raise RuntimeError(f"NLP Asset failed to load: {self._load_error}")

        assert self._embedder is not None and self._vector_store is not None

        start_time = time.perf_counter()

        # 1. Embed query
        query_vector = self._embedder.embed_query(req.query)

        # 2. Retrieve paper hits
        hits: list[PaperHit] = self._vector_store.query(query_vector, top_k=req.paper_top_k)

        # 3. Build ranking strategy & aggregate hits by author
        week4_cfg = Week4Config(
            ranking_strategy=req.strategy,  # type: ignore[arg-type]
            strategy_top_k=3,
        )
        strategy = create_strategy(week4_cfg)
        aggregator = Aggregator(strategy)
        ranked_scores: list[ResearcherScore] = aggregator.aggregate(hits, top_k=req.top_k * 2)

        # 4. Calibrate confidence scores
        calibrator = ConfidenceCalibrator(strategy=req.strategy)
        calibrated_confs = calibrator.calibrate_ranking(ranked_scores)

        # 5. Build structured recommendations and filter by min_confidence
        recommendations: list[ResearcherRecommendation] = []
        rank_counter = 1
        for res_score, conf in zip(ranked_scores, calibrated_confs, strict=True):
            if conf.score < req.min_confidence:
                continue

            evidence_papers: list[PaperEvidence] = []
            if req.include_evidence:
                for hit in res_score.evidence[:3]:
                    year = hit.paper.metadata.get("publication_year", "N/A")
                    evidence_papers.append(
                        PaperEvidence(
                            paper_id=hit.paper.paper_id,
                            title=hit.paper.title,
                            year=year,
                            similarity=hit.score,
                            authors_count=len(hit.paper.authors),
                        )
                    )

            rec = ResearcherRecommendation(
                rank=rank_counter,
                author_id=res_score.author.author_id,
                name=res_score.author.name,
                score=res_score.score,
                confidence=conf,
                matched_papers=res_score.matched_papers,
                evidence=evidence_papers,
                topics=self._topics_from_evidence(res_score.evidence),
            )
            recommendations.append(rec)
            rank_counter += 1
            if len(recommendations) >= req.top_k:
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        model_info = ModelInfo(
            name=self.config.model_name,
            strategy=req.strategy,
            backend=self._backend_type,
            embedding_dim=384,
            ranking_explanation=self._ranking_explanation(req.strategy),
        )

        return RecommendResponse(
            query=req.query,
            execution_time_ms=elapsed_ms,
            total_results=len(recommendations),
            model_info=model_info,
            results=recommendations,
        )

    def batch_recommend(
        self, requests: Sequence[RecommendRequest | Mapping[str, Any] | str]
    ) -> list[RecommendResponse]:
        """Process multiple recommendation queries efficiently."""
        return [self.recommend(req) for req in requests]

    def get_metadata(self) -> ModelInfo:
        return ModelInfo(
            name=self.config.model_name,
            strategy=self.config.default_strategy,
            backend=self._backend_type,
            embedding_dim=384,
            ranking_explanation=self._ranking_explanation(self.config.default_strategy),
        )

    @staticmethod
    def _topics_from_evidence(evidence: Sequence[PaperHit], *, limit: int = 8) -> list[str]:
        """Return distinct topic labels carried by the supporting papers."""
        topics: list[str] = []
        seen: set[str] = set()
        for hit in evidence:
            raw = str(hit.paper.metadata.get("topics", ""))
            for topic in raw.split(","):
                normalized = topic.strip()
                if normalized and normalized.lower() not in seen:
                    seen.add(normalized.lower())
                    topics.append(normalized)
                    if len(topics) >= limit:
                        return topics
        return topics

    @staticmethod
    def _ranking_explanation(strategy: str) -> str:
        explanations = {
            "top_k_average_author_discount": (
                "Ranks each researcher by the mean of their three strongest matching papers, "
                "discounting papers with many co-authors by sqrt(author count). Publication year, "
                "teaching status, and mentor availability are not ranking signals."
            ),
            "top_k_average": (
                "Ranks each researcher by the mean of their three strongest matching papers. "
                "Publication year, teaching status, and mentor availability are not ranking signals."
            ),
            "best_paper": (
                "Ranks each researcher by their single strongest matching paper. Publication year, "
                "teaching status, and mentor availability are not ranking signals."
            ),
            "average": (
                "Ranks each researcher by the average similarity of retrieved papers. Publication year, "
                "teaching status, and mentor availability are not ranking signals."
            ),
        }
        return explanations.get(strategy, "Ranks publication similarity; teaching status is not verified.")

    def health_check(self) -> dict[str, Any]:
        """Report component health, readiness, and storage statistics."""
        return {
            "status": "ready" if self.is_ready else ("failed" if self._load_error else "unloaded"),
            "ready": self.is_ready,
            "backend": self._backend_type,
            "model_name": self.config.model_name,
            "default_strategy": self.config.default_strategy,
            "error": self._load_error,
        }
