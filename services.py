"""Data preparation for the web interface.

Everything the interface shows is prepared here as plain Python data. This
module renders no HTML and imports no web framework, so it can be used and
tested without a server. Retrieval itself stays in
``src.week2.pipeline.ResearchDiscoveryPipeline`` and is called unchanged.
"""

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.week6.confidence import ConfidenceCalibrator
from src.week6.laya_classifier import get_shared_laya_classifier
from src.week6.reasoning import get_shared_reasoning_engine
from src.week6.schemas import CitationSource, ClassifyPaperInput, QueryClassification, ReasoningRequest, ReasoningResponse

PROJECT_ROOT = Path(__file__).resolve().parent
CALCULATED_DIR = PROJECT_ROOT / "Data" / "processed" / "calculated"
WEEK5_PHASE2_DIR = PROJECT_ROOT / "src" / "week5" / "artifacts" / "phase2"
WEEK5_FAIR_DIR = WEEK5_PHASE2_DIR / "fair_evaluation" / "20261008T050903Z"
WEEK5_VARIANT_DIR = WEEK5_PHASE2_DIR / "improved_variant" / "20261008T050358Z"
DEFAULT_GROUP6_URL = os.environ.get("GROUP6_URL", "http://127.0.0.1:5000")
GROUP6_HEALTH_TIMEOUT = 2.0
MAX_TOPIC_BADGES = 6

# Retrieval modes accepted by ResearchDiscoveryPipeline.search, with display labels.
MODE_LABELS: dict[str, str] = {
    "paper_discount": "Paper-Level + Co-Author Discount (Week 5/6)",
    "semantic": "Dense Profile Semantic (Sentence-BERT)",
    "tfidf": "TF-IDF (Keyword Baseline)",
    "hybrid": "Hybrid (Semantic + TF-IDF)",
}
DEFAULT_MODE = "paper_discount"
TOP_K_MIN, TOP_K_MAX, TOP_K_STEP, DEFAULT_TOP_K = 5, 25, 5, 10
ALPHA_MIN, ALPHA_MAX, ALPHA_STEP, DEFAULT_ALPHA = 0.0, 1.0, 0.05, 0.7

MIN_PAPERS_OPTIONS: tuple[tuple[int, str], ...] = (
    (1, "≥ 1 paper (All)"),
    (2, "≥ 2 papers"),
    (3, "≥ 3 papers"),
    (5, "≥ 5 papers (Established)"),
)
VALID_MIN_PAPERS = frozenset(val for val, _ in MIN_PAPERS_OPTIONS)

SINCE_YEAR_OPTIONS: tuple[tuple[str, str], ...] = (
    ("", "Any publication year"),
    ("2022", "2022+ (Active)"),
    ("2024", "2024+ (Recent)"),
)
VALID_SINCE_YEARS = frozenset({2022, 2024})
MAX_CLUSTER_CHIPS = 6

DOMAIN_CLUSTERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "NLP & Language Models",
        (
            "natural language",
            "nlp",
            "language model",
            "machine translation",
            "translation",
            "speech",
            "sentiment",
            "topic model",
            "information extraction",
            "information retrieval",
            "question answering",
            "named entity",
            "summarization",
            "cross-lingual",
            "multilingual",
            "text classification",
        ),
    ),
    (
        "Machine Learning & Deep Learning",
        (
            "machine learning",
            "deep learning",
            "neural network",
            "computer vision",
            "reinforcement learning",
            "graph neural",
            "knowledge graph",
            "representation learning",
            "multimodal",
        ),
    ),
    (
        "Electrical Power & Energy Systems",
        (
            "power transformer",
            "power system",
            "high voltage",
            "hvdc",
            "microgrid",
            "islanding",
            "electrical fault",
            "battery",
            "energy storage",
            "photovoltaic",
            "insulation",
            "dielectric",
            "power transmission",
        ),
    ),
    (
        "Biomedical & Healthcare",
        (
            "biomedical",
            "healthcare",
            "medical",
            "clinical",
            "drug",
            "tuberculosis",
            "disease",
            "genom",
            "protein",
            "health information",
        ),
    ),
    (
        "Hardware, VLSI & Systems",
        (
            "vlsi",
            "parallel computing",
            "embedded",
            "semiconductor",
            "hardware",
            "wireless",
            "iot",
            "network security",
            "data compression",
        ),
    ),
    (
        "Quantum & Physical Sciences",
        (
            "quantum",
            "optics",
            "photon",
            "condensed matter",
            "magnetic",
            "seism",
            "colloid",
        ),
    ),
)

QUERY_PRESETS: tuple[tuple[str, str], ...] = (
    ("NLP & Information Extraction", "Application of Large Language Models to information extraction"),
    ("Transformer Classification", "transformer based text classification"),
    ("Knowledge Graph Reasoning", "knowledge graph embeddings and reasoning"),
    ("Cross-Lingual Retrieval", "cross-lingual information retrieval and neural search"),
)

COMPARISON_DEFAULT_QUERY = "transformer based text classification"
COMPARISON_TOP_K = 5
COMPARISON_MODES = ("paper_discount", "semantic", "tfidf")


@dataclass(frozen=True)
class EvaluationTable:
    """A benchmark or audit CSV file shown on the evaluation page."""

    key: str
    title: str
    path: Path


EVALUATION_TABLES: tuple[EvaluationTable, ...] = (
    EvaluationTable(
        "fair_summary",
        "Week 5 Fair Evaluation Summary (214 Pooled Judgments, 0-3 Graded Rubric)",
        WEEK5_FAIR_DIR / "summary.csv",
    ),
    EvaluationTable(
        "coauthor_diversity",
        "Co-Author Concentration and Diversity Impact (Top-10 per Query)",
        WEEK5_VARIANT_DIR / "coauthor_concentration.csv",
    ),
    EvaluationTable(
        "fair_per_query",
        "Week 5 Fair Evaluation by Query (All 6 Pooled Systems)",
        WEEK5_FAIR_DIR / "per_query.csv",
    ),
    EvaluationTable(
        "old_label_audit",
        "Biased Single-System Pool Comparison (Old TF-IDF Labels on Frozen Runs)",
        WEEK5_VARIANT_DIR / "old_label_summary.csv",
    ),
    EvaluationTable(
        "model_comparison",
        "Legacy Week 3 Summary (Biased Single-System TF-IDF Pool)",
        CALCULATED_DIR / "model_comparison.csv",
    ),
    EvaluationTable(
        "baseline_metrics",
        "Legacy TF-IDF Baseline by Query (Self-Labeled Pool)",
        CALCULATED_DIR / "baseline_metrics.csv",
    ),
    EvaluationTable(
        "semantic_metrics",
        "Legacy Sentence Transformers by Query",
        CALCULATED_DIR / "semantic_metrics.csv",
    ),
)


@dataclass(frozen=True)
class SearchSettings:
    """Validated search configuration and collaboration filters."""

    mode: str = DEFAULT_MODE
    top_k: int = DEFAULT_TOP_K
    alpha: float = DEFAULT_ALPHA
    institution: str = ""
    min_papers: int = 1
    since_year: int | None = None
    topic: str = ""
    reason: bool = False
    reason_provider: str = "auto"

    @property
    def mode_label(self) -> str:
        return MODE_LABELS[self.mode]

    @property
    def effective_alpha(self) -> float:
        """Alpha only applies to hybrid mode; other modes always pass the default."""
        return self.alpha if self.mode == "hybrid" else DEFAULT_ALPHA

    @property
    def has_active_filters(self) -> bool:
        return bool(
            self.institution
            or self.min_papers > 1
            or self.since_year is not None
            or self.topic
        )


@dataclass(frozen=True)
class Stat:
    """A dataset figure: counts are integers, formatted by the template."""

    label: str
    value: int | str


@dataclass(frozen=True)
class PaperMatch:
    """One supporting publication for a ranked researcher."""

    title: str
    year: str
    score: float
    work_url: str | None = None
    doi_url: str | None = None
    doi_label: str | None = None
    abstract: str = ""
    authors_count: int = 1
    paper_type: str = ""
    paper_type_label: str = ""
    paper_type_confidence: float = 0.0
    relevance_label: str = ""
    is_scholarly: bool = True


@dataclass(frozen=True)
class ResearcherResult:
    """One ranked researcher, ready for display."""

    rank: int
    name: str
    author_id: str
    author_url: str | None
    similarity_score: float
    n_papers: int
    paper_title: str
    paper_year: str
    paper_score: float
    topics: tuple[str, ...]
    matched_papers: int = 1
    confidence_score: float = 0.5
    confidence_level: str = "MEDIUM"
    confidence_explanation: str = ""
    rank_delta: str | None = None
    institution: str = ""
    top_papers: tuple[PaperMatch, ...] = ()
    match_rationale: str = ""


@dataclass(frozen=True)
class TopicCluster:
    """A dynamic topic or sub-field filter chip derived from retrieved results."""

    label: str
    count: int
    active: bool = False


@dataclass(frozen=True)
class SearchOutcome:
    settings: SearchSettings
    results: tuple[ResearcherResult, ...]
    elapsed_seconds: float
    topic_clusters: tuple[TopicCluster, ...] = ()
    total_before_filters: int = 0
    query_classification: QueryClassification | None = None
    reasoning: ReasoningResponse | None = None


@dataclass(frozen=True)
class ComparisonColumn:
    mode: str
    title: str
    results: tuple[ResearcherResult, ...]
    distinct_papers: int = 0


@dataclass(frozen=True)
class TableData:
    """A CSV file as display strings, exactly as written in the file."""

    table: EvaluationTable
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


COMPARISON_TITLES: dict[str, str] = {
    "paper_discount": "Paper-Level + Author Discount (Week 5/6)",
    "semantic": "Sentence Transformers (Dense Semantic)",
    "tfidf": "TF-IDF Baseline (Sparse Keyword)",
}


def parse_search_settings(args: Mapping[str, str]) -> tuple[SearchSettings, list[str]]:
    """Read settings from request parameters; invalid values fall back to defaults with an error."""
    errors: list[str] = []
    mode = args.get("mode", DEFAULT_MODE)
    if mode not in MODE_LABELS:
        errors.append(f"Unknown retrieval mode {mode!r}.")
        mode = DEFAULT_MODE

    top_k = DEFAULT_TOP_K
    raw_top_k = args.get("top_k")
    if raw_top_k is not None:
        try:
            top_k = int(raw_top_k)
            if not (TOP_K_MIN <= top_k <= TOP_K_MAX and top_k % TOP_K_STEP == 0):
                raise ValueError
        except ValueError:
            errors.append(f"Number of researchers must be one of {TOP_K_MIN}–{TOP_K_MAX} in steps of {TOP_K_STEP}.")
            top_k = DEFAULT_TOP_K

    alpha = DEFAULT_ALPHA
    raw_alpha = args.get("alpha")
    if raw_alpha is not None:
        try:
            alpha = float(raw_alpha)
            if not ALPHA_MIN <= alpha <= ALPHA_MAX:
                raise ValueError
        except ValueError:
            errors.append(f"Semantic weight must be between {ALPHA_MIN} and {ALPHA_MAX}.")
            alpha = DEFAULT_ALPHA

    institution = " ".join(args.get("institution", "").strip().split())[:120]

    min_papers = 1
    raw_min_papers = args.get("min_papers")
    if raw_min_papers is not None and raw_min_papers.strip() != "":
        try:
            min_papers = int(raw_min_papers)
            if min_papers not in VALID_MIN_PAPERS:
                raise ValueError
        except ValueError:
            errors.append("Minimum indexed papers must be one of 1, 2, 3, or 5.")
            min_papers = 1

    since_year: int | None = None
    raw_since = args.get("since_year")
    if raw_since is not None and raw_since.strip().lower() not in {"", "any"}:
        try:
            parsed_year = int(raw_since)
            if parsed_year not in VALID_SINCE_YEARS:
                raise ValueError
            since_year = parsed_year
        except ValueError:
            errors.append("Published since filter must be Any, 2022, or 2024.")
            since_year = None

    topic = " ".join(args.get("topic", "").strip().split())[:100]
    reason = str(args.get("reason", "")).strip().lower() in {"1", "true", "yes"}
    reason_provider = str(args.get("reason_provider", "auto")).strip().lower()
    if reason_provider not in {"auto", "groq", "gemini"}:
        reason_provider = "auto"

    return (
        SearchSettings(
            mode=mode,
            top_k=top_k,
            alpha=alpha,
            institution=institution,
            min_papers=min_papers,
            since_year=since_year,
            topic=topic,
            reason=reason,
            reason_provider=reason_provider,
        ),
        errors,
    )


def _safe_http_url(value: Any) -> str | None:
    text = str(value or "").strip()
    return text if text.startswith(("https://", "http://")) else None


def _normalize_doi(raw_doi: Any) -> tuple[str | None, str | None]:
    """Return a validated ``https://doi.org/...`` URL and compact DOI identifier."""
    text = str(raw_doi or "").strip()
    if not text or text.lower() in {"nan", "none", "n/a", "unknown"}:
        return None, None
    lowered = text.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/"):
        if lowered.startswith(prefix):
            ident = text[len(prefix) :].strip()
            if ident.startswith("10."):
                return f"https://doi.org/{ident}", ident
            return None, None
    if text.startswith("10.") and "/" in text and " " not in text:
        return f"https://doi.org/{text}", text
    return None, None


def _clean_topics(raw_topics: Sequence[Any], limit: int = MAX_TOPIC_BADGES) -> tuple[str, ...]:
    """Split topic strings on '|' and ',', deduplicate case-insensitively, and cap at ``limit``."""
    cleaned: list[str] = []
    seen: set[str] = set()
    ignored = {"", "nan", "none", "n/a", "unknown"}
    for item in raw_topics:
        for seg in re.split(r"[|,]", str(item)):
            norm = " ".join(seg.strip().split())
            key = norm.lower()
            if key not in ignored and key not in seen:
                seen.add(key)
                cleaned.append(norm)
                if len(cleaned) >= limit:
                    return tuple(cleaned)
    return tuple(cleaned)


def _normalize_institution(raw_inst: Any) -> str:
    text = " ".join(str(raw_inst or "").strip().split())
    if text.lower() in {"", "nan", "none", "n/a", "unknown"}:
        return ""
    parts = [p.strip() for p in re.split(r"[|;]", text) if p.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part.lower() not in seen:
            seen.add(part.lower())
            deduped.append(part)
    return " · ".join(deduped[:2])


def _to_paper_match(paper: Mapping[str, Any]) -> PaperMatch:
    title = str(paper.get("title", "N/A")).strip() or "N/A"
    year = str(paper.get("year", "N/A")).strip() or "N/A"
    score = float(paper.get("paper_score", paper.get("score", 0.0)))
    work_url = _safe_http_url(paper.get("work_id"))
    doi_url, doi_label = _normalize_doi(paper.get("doi"))
    abstract = " ".join(str(paper.get("abstract", "")).strip().split())
    if abstract.lower() in {"nan", "none", "n/a"}:
        abstract = ""
    authors_count = max(1, int(paper.get("authors_count", 1)))
    return PaperMatch(
        title=title,
        year=year,
        score=score,
        work_url=work_url,
        doi_url=doi_url,
        doi_label=doi_label,
        abstract=abstract,
        authors_count=authors_count,
    )


def _build_match_rationale(
    mode: str,
    sim_score: float,
    n_papers: int,
    matched_papers: int,
    primary_paper: PaperMatch | None,
) -> str:
    best_score = primary_paper.score if primary_paper is not None else 0.0
    coauthors = primary_paper.authors_count if primary_paper is not None else 1
    if mode == "paper_discount":
        return (
            f"Aggregated across {matched_papers} retrieved publication(s) using top-3 paper similarity "
            f"discounted by 1/√co-authors (lead paper raw similarity {best_score:.4f}, "
            f"{coauthors} author{'s' if coauthors != 1 else ''})."
        )
    if mode == "semantic":
        return (
            f"Dense Sentence-BERT profile cosine similarity {sim_score:.4f} across {n_papers} indexed "
            f"publication(s); strongest individual paper match {best_score:.4f}."
        )
    if mode == "hybrid":
        return (
            f"Blended dense Sentence-BERT + sparse TF-IDF profile score {sim_score:.4f} across "
            f"{n_papers} indexed publication(s); strongest paper match {best_score:.4f}."
        )
    return (
        f"Sparse unigram/bigram TF-IDF profile overlap {sim_score:.4f} across {n_papers} indexed "
        f"publication(s); strongest lexical paper match {best_score:.4f}."
    )


def to_researcher_result(
    raw: Mapping[str, Any],
    *,
    calibrator: ConfidenceCalibrator | None = None,
    next_score: float | None = None,
    rank_delta: str | None = None,
) -> ResearcherResult:
    """Convert one ``pipeline.search`` result into display data with calibrated confidence."""
    paper = raw["relevant_paper"]
    author_id = str(raw["author_id"])
    rank = int(raw["rank"])
    sim_score = float(raw["similarity_score"])
    n_papers = int(raw["n_papers"])
    mode = str(raw.get("mode", DEFAULT_MODE))
    matched_papers = max(1, int(raw.get("matched_papers", paper.get("matched_papers", 1))))
    cal = calibrator or ConfidenceCalibrator(strategy=mode)
    conf = cal.calibrate(
        raw_score=sim_score,
        rank=rank,
        matched_papers=matched_papers,
        next_score=next_score,
    )

    raw_matching = raw.get("matching_papers") or paper.get("top_papers") or [paper]
    parsed_papers = tuple(
        pm for pm in (_to_paper_match(p) for p in raw_matching[:3]) if pm.title.upper() != "N/A"
    )
    primary_match = parsed_papers[0] if parsed_papers else _to_paper_match(paper)
    institution = _normalize_institution(raw.get("institution_name") or paper.get("institutions"))
    rationale = _build_match_rationale(mode, sim_score, n_papers, matched_papers, primary_match)

    return ResearcherResult(
        rank=rank,
        name=str(raw["author_name"]),
        author_id=author_id,
        author_url=_safe_http_url(author_id),
        similarity_score=sim_score,
        n_papers=n_papers,
        paper_title=str(paper["title"]),
        paper_year=str(paper.get("year", "N/A")),
        paper_score=float(paper.get("paper_score", 0.0)),
        topics=_clean_topics(raw["related_topics"]),
        matched_papers=matched_papers,
        confidence_score=conf.score,
        confidence_level=conf.level,
        confidence_explanation=conf.explanation,
        rank_delta=rank_delta,
        institution=institution,
        top_papers=parsed_papers,
        match_rationale=rationale,
    )


def _attach_laya_classifications(
    results: Sequence[ResearcherResult],
    query: str,
) -> tuple[ResearcherResult, ...]:
    """Batch-classify supporting publications on researcher cards using Laya System-1."""
    if not results:
        return ()

    inputs: list[ClassifyPaperInput] = []
    positions: list[tuple[int, int]] = []
    for r_idx, res in enumerate(results):
        topics_str = ", ".join(res.topics)
        for p_idx, pm in enumerate(res.top_papers):
            if not pm.title or pm.title.upper() == "N/A":
                continue
            inputs.append(
                ClassifyPaperInput(
                    title=pm.title,
                    abstract=pm.abstract,
                    topics=topics_str,
                    paper_id=pm.work_url or f"R{res.rank}_P{p_idx + 1}",
                    similarity=pm.score,
                )
            )
            positions.append((r_idx, p_idx))

    if not inputs:
        return tuple(results)

    classified = get_shared_laya_classifier().classify_papers(inputs, query=query)
    paper_lists: list[list[PaperMatch]] = [list(r.top_papers) for r in results]
    for (r_idx, p_idx), cls_res in zip(positions, classified, strict=True):
        old_pm = paper_lists[r_idx][p_idx]
        paper_lists[r_idx][p_idx] = replace(
            old_pm,
            paper_type=cls_res.paper_type,
            paper_type_label=cls_res.paper_type_label,
            paper_type_confidence=cls_res.type_confidence,
            relevance_label=cls_res.relevance_label,
            is_scholarly=cls_res.is_scholarly,
        )

    return tuple(
        replace(res, top_papers=tuple(paper_lists[r_idx]))
        for r_idx, res in enumerate(results)
    )


def _convert_ranked_results(
    raw_list: Sequence[Mapping[str, Any]],
    mode: str,
    *,
    baseline_ranks: Mapping[str, int] | None = None,
    query: str = "",
) -> tuple[ResearcherResult, ...]:
    """Convert a ranked list with pairwise score margins, rank deltas, and Laya paper labels."""
    calibrator = ConfidenceCalibrator(strategy=mode)
    converted: list[ResearcherResult] = []
    total = len(raw_list)
    for idx, raw in enumerate(raw_list):
        next_score = float(raw_list[idx + 1]["similarity_score"]) if idx + 1 < total else None
        delta: str | None = None
        if baseline_ranks is not None:
            author_id = str(raw["author_id"])
            curr_rank = int(raw["rank"])
            if author_id in baseline_ranks:
                diff = baseline_ranks[author_id] - curr_rank
                if diff > 0:
                    delta = f"↑{diff} vs TF-IDF"
                elif diff < 0:
                    delta = f"↓{abs(diff)} vs TF-IDF"
                else:
                    delta = "= TF-IDF"
            else:
                delta = f"New in Top {COMPARISON_TOP_K}"
        converted.append(
            to_researcher_result(raw, calibrator=calibrator, next_score=next_score, rank_delta=delta)
        )
    return _attach_laya_classifications(converted, query=query)


def _count_distinct_papers(results: Sequence[ResearcherResult]) -> int:
    """Count distinct primary evidence paper titles across a ranked result slice."""
    return len(
        {
            r.paper_title.strip().lower()
            for r in results
            if r.paper_title.strip() and r.paper_title.strip().upper() != "N/A"
        }
    )


def _parse_int_year(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "n/a", "unknown"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _raw_candidate_haystack(raw: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    topics = _clean_topics(raw.get("related_topics", ()), limit=12)
    paper = raw.get("relevant_paper", {})
    title = str(paper.get("title", ""))
    joined = " | ".join((*topics, title)).lower()
    return topics, joined


def _raw_matches_topic(raw: Mapping[str, Any], target_topic: str) -> bool:
    needle = target_topic.strip().lower()
    if not needle:
        return True
    topics, joined = _raw_candidate_haystack(raw)
    for cluster_label, keywords in DOMAIN_CLUSTERS:
        if cluster_label.lower() == needle:
            return any(kw in joined for kw in keywords)
    return any(t.lower() == needle or needle in t.lower() for t in topics)


def _raw_matches_filters(
    raw: Mapping[str, Any],
    settings: SearchSettings,
    *,
    include_topic: bool = True,
) -> bool:
    if settings.min_papers > 1 and int(raw.get("n_papers", 0)) < settings.min_papers:
        return False

    if settings.institution:
        paper = raw.get("relevant_paper", {})
        inst_haystack = " ; ".join(
            (
                str(raw.get("institution_name", "")),
                str(paper.get("institutions", "")),
            )
        ).lower()
        if settings.institution.lower() not in inst_haystack:
            return False

    if settings.since_year is not None:
        paper = raw.get("relevant_paper", {})
        candidate_papers = raw.get("matching_papers") or paper.get("top_papers") or [paper]
        years = [_parse_int_year(p.get("year")) for p in candidate_papers]
        if not any(y is not None and y >= settings.since_year for y in years):
            return False

    if include_topic and settings.topic and not _raw_matches_topic(raw, settings.topic):
        return False

    return True


def _build_topic_clusters_from_raw(
    raw_list: Sequence[Mapping[str, Any]],
    active_topic: str = "",
) -> tuple[TopicCluster, ...]:
    """Derive disambiguation sub-field and topic chips from a candidate slice."""
    if not raw_list:
        if active_topic:
            return (TopicCluster(label=active_topic, count=0, active=True),)
        return ()

    active_lower = active_topic.strip().lower()
    domain_counts: list[tuple[str, int]] = []
    for cluster_label, keywords in DOMAIN_CLUSTERS:
        matched = sum(
            1
            for raw in raw_list
            if any(kw in _raw_candidate_haystack(raw)[1] for kw in keywords)
        )
        if matched > 0:
            domain_counts.append((cluster_label, matched))

    clusters: list[TopicCluster] = []
    seen_labels: set[str] = set()

    # When a query spans 2+ distinct high-level sub-fields (e.g., "transformer"),
    # lead with those domain clusters for one-click disambiguation.
    if len(domain_counts) >= 2:
        domain_counts.sort(key=lambda item: (-item[1], item[0]))
        for label, count in domain_counts[:MAX_CLUSTER_CHIPS]:
            seen_labels.add(label.lower())
            clusters.append(
                TopicCluster(label=label, count=count, active=(label.lower() == active_lower))
            )

    topic_freq: dict[str, int] = {}
    canonical_label: dict[str, str] = {}
    for raw in raw_list:
        for topic in _clean_topics(raw.get("related_topics", ()), limit=MAX_TOPIC_BADGES):
            key = topic.lower()
            if key in seen_labels:
                continue
            canonical_label.setdefault(key, topic)
            topic_freq[key] = topic_freq.get(key, 0) + 1

    sorted_topics = sorted(topic_freq.items(), key=lambda item: (-item[1], canonical_label[item[0]]))
    for key, count in sorted_topics:
        if len(clusters) >= MAX_CLUSTER_CHIPS:
            break
        label = canonical_label[key]
        seen_labels.add(key)
        clusters.append(TopicCluster(label=label, count=count, active=(key == active_lower)))

    if active_topic and active_lower not in seen_labels:
        active_count = sum(1 for raw in raw_list if _raw_matches_topic(raw, active_topic))
        clusters.insert(0, TopicCluster(label=active_topic, count=active_count, active=True))
        clusters = clusters[:MAX_CLUSTER_CHIPS]

    return tuple(clusters)


def institution_suggestions(pipeline: Any, limit: int = 30) -> tuple[str, ...]:
    """Return the most frequent institution names in the corpus for search autocomplete."""
    cached = getattr(pipeline, "_cached_top_institutions", None)
    if cached is not None:
        return cached
    profiles_df = getattr(pipeline, "profiles_df", None)
    if profiles_df is None or "institution_name" not in profiles_df.columns:
        return ()
    counts: dict[str, int] = {}
    for raw_val in profiles_df["institution_name"]:
        for part in str(raw_val).split(";"):
            clean = part.strip()
            if clean and clean.lower() not in {"nan", "none", "unknown"}:
                counts[clean] = counts.get(clean, 0) + 1
    top = tuple(name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit])
    try:
        setattr(pipeline, "_cached_top_institutions", top)
    except Exception:  # noqa: BLE001
        pass
    return top


def synthesize_reasoning_for_results(
    query: str,
    results: Sequence[ResearcherResult],
    provider: str = "auto",
) -> ReasoningResponse:
    """Synthesize citation-backed grounded answer from top researcher publications."""
    citations: list[CitationSource] = []
    seen_ids: set[str] = set()
    for r in results:
        for pm in r.top_papers:
            ident = pm.doi_url or pm.work_url or pm.title
            if ident not in seen_ids and len(citations) < 8 and pm.title.upper() != "N/A":
                seen_ids.add(ident)
                citations.append(
                    CitationSource(
                        paper_index=len(citations) + 1,
                        title=pm.title,
                        paper_id=ident,
                        year=pm.year,
                        abstract=pm.abstract,
                        doi=pm.doi_label,
                        authors=(r.name,),
                        similarity=pm.score,
                    )
                )
    return get_shared_reasoning_engine().reason(
        ReasoningRequest(query=query, papers=tuple(citations), provider=provider)
    )


def run_search(pipeline: Any, query: str, settings: SearchSettings) -> SearchOutcome:
    """Rank researchers for ``query`` with the chosen settings and collaboration filters."""
    query = query.strip()
    start = time.time()
    query_cls = get_shared_laya_classifier().classify_query(query) if query else None
    if not settings.has_active_filters:
        raw = pipeline.search(
            query=query,
            top_k=settings.top_k,
            mode=settings.mode,
            alpha=settings.effective_alpha,
        )
        results = _convert_ranked_results(raw, settings.mode, query=query)
        elapsed = time.time() - start
        clusters = _build_topic_clusters_from_raw(raw, active_topic="")
        reasoning = (
            synthesize_reasoning_for_results(query, results, provider=settings.reason_provider)
            if settings.reason and results
            else None
        )
        return SearchOutcome(
            settings=settings,
            results=results,
            elapsed_seconds=elapsed,
            topic_clusters=clusters,
            total_before_filters=len(results),
            query_classification=query_cls,
            reasoning=reasoning,
        )

    candidate_k = min(max(settings.top_k * 8, 120), 250)
    raw_pool = pipeline.search(
        query=query,
        top_k=candidate_k,
        mode=settings.mode,
        alpha=settings.effective_alpha,
    )
    pre_topic = [r for r in raw_pool if _raw_matches_filters(r, settings, include_topic=False)]
    clusters = _build_topic_clusters_from_raw(pre_topic, active_topic=settings.topic)
    filtered = (
        [r for r in pre_topic if _raw_matches_topic(r, settings.topic)]
        if settings.topic
        else pre_topic
    )
    renumbered = [
        {**dict(item), "rank": idx}
        for idx, item in enumerate(filtered[: settings.top_k], start=1)
    ]
    results = _convert_ranked_results(renumbered, settings.mode, query=query)
    elapsed = time.time() - start
    reasoning = (
        synthesize_reasoning_for_results(query, results, provider=settings.reason_provider)
        if settings.reason and results
        else None
    )
    return SearchOutcome(
        settings=settings,
        results=results,
        elapsed_seconds=elapsed,
        topic_clusters=clusters,
        total_before_filters=len(raw_pool),
        query_classification=query_cls,
        reasoning=reasoning,
    )


def run_comparison(pipeline: Any, query: str) -> list[ComparisonColumn]:
    """Top researchers for ``query`` across Paper-Level, Dense Semantic, and TF-IDF modes."""
    raw_by_mode = {
        mode: pipeline.search(query=query, top_k=COMPARISON_TOP_K, mode=mode)
        for mode in COMPARISON_MODES
    }
    tfidf_ranks = {str(r["author_id"]): int(r["rank"]) for r in raw_by_mode.get("tfidf", ())}
    columns: list[ComparisonColumn] = []
    for mode in COMPARISON_MODES:
        results = _convert_ranked_results(
            raw_by_mode[mode],
            mode,
            baseline_ranks=tfidf_ranks if mode != "tfidf" else None,
            query=query,
        )
        columns.append(
            ComparisonColumn(
                mode=mode,
                title=COMPARISON_TITLES[mode],
                results=results,
                distinct_papers=_count_distinct_papers(results),
            )
        )
    return columns


def dataset_stats(pipeline: Any) -> list[Stat]:
    """Dataset size and model, as shown alongside every page."""
    stats = []
    if pipeline.profiles_df is not None:
        stats.append(Stat("Indexed OpenAlex Researchers", len(pipeline.profiles_df)))
    if pipeline.papers_df is not None:
        stats.append(Stat("Indexed Academic Papers", len(pipeline.papers_df)))
    stats.append(Stat("Embedding Model", pipeline.model_name))
    return stats


def group6_health(base_url: str = DEFAULT_GROUP6_URL) -> dict[str, Any]:
    """Probe the Group 6 integration UI without making it a Group 7 dependency.

    The Group 6 UI exposes ``GET /health`` as a local liveness endpoint. This
    helper deliberately distinguishes a refused connection from a reachable
    but incompatible service, which makes port/process mistakes visible in the
    Group 7 dashboard.
    """
    url = base_url.rstrip("/")
    endpoint = url + "/health"
    request = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=GROUP6_HEALTH_TIMEOUT) as response:
            payload = json.load(response)
            status = payload.get("status", "unknown")
            service = payload.get("service", "unknown")
            identity_ok = service == "group6-integration-ui"
            healthy = identity_ok and status in ("ready", "ok")
            state = status if status in ("ready", "ok", "starting") else "unhealthy"
            if not identity_ok:
                state = "incompatible"
            return {
                "state": state,
                "status": status,
                "reachable": True,
                "healthy": healthy,
                "http_status": getattr(response, "status", 200),
                "url": url,
                "service": service,
                "detail": "Group 6 integration UI is reachable." if healthy else "The configured endpoint is not the expected Group 6 integration UI health service." if not identity_ok else "Group 6 integration UI is not ready.",
            }
    except urllib.error.HTTPError as exc:
        return {
            "state": "incompatible" if exc.code < 500 else "unhealthy",
            "status": "http_error",
            "reachable": True,
            "healthy": False,
            "http_status": exc.code,
            "url": url,
            "service": "unknown",
            "detail": "A service answered, but it does not expose the Group 6 /health contract." if exc.code == 404 else "Group 6 returned an unhealthy HTTP status.",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "state": "unreachable",
            "status": "unreachable",
            "reachable": False,
            "healthy": False,
            "http_status": None,
            "url": url,
            "service": "unknown",
            "detail": f"Could not connect to Group 6 ({type(exc).__name__}).",
        }
    except (ValueError, json.JSONDecodeError):
        return {
            "state": "incompatible",
            "status": "invalid_response",
            "reachable": True,
            "healthy": False,
            "http_status": None,
            "url": url,
            "service": "unknown",
            "detail": "A service answered, but its Group 6 health response was not valid JSON.",
        }


def evaluation_table(key: str) -> EvaluationTable | None:
    return next((table for table in EVALUATION_TABLES if table.key == key), None)


def load_table(table: EvaluationTable) -> TableData | None:
    """Read a metrics CSV as strings; ``None`` when the file does not exist."""
    if not table.path.exists():
        return None
    frame = pd.read_csv(table.path, dtype=str, keep_default_na=False)
    return TableData(table, tuple(frame.columns), tuple(tuple(row) for row in frame.itertuples(index=False)))


@dataclass
class PipelineLoader:
    """Loads the retrieval pipeline once and shares it between requests.

    Loading takes tens of seconds, so it can run in a background thread while
    the interface reports progress. Searches are serialised with a lock
    because the embedding model is not guaranteed to be thread-safe.
    """

    factory: Callable[[], Any]
    on_ready: Callable[[Any], None] | None = None
    pipeline: Any = None
    error: str | None = None
    _started: bool = False
    _ready: threading.Event = field(default_factory=threading.Event)
    _start_lock: threading.Lock = field(default_factory=threading.Lock)
    search_lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self, background: bool = True) -> None:
        """Begin loading once; later calls do nothing."""
        with self._start_lock:
            if self._started:
                return
            self._started = True
        if background:
            threading.Thread(target=self._load, name="pipeline-loader", daemon=True).start()
        else:
            self._load()

    def _load(self) -> None:
        try:
            self.pipeline = self.factory()
        except Exception as exc:  # noqa: BLE001
            # Deliberately broad. Real load failures include classes that derive
            # directly from Exception (corrupt .npz cache: BadZipFile; corrupt model
            # weights: SafetensorError). An uncaught error would end this thread with
            # no error recorded, the app would report "ready" without a pipeline,
            # and every page would fail. The message is shown on the error page.
            self.error = str(exc) or exc.__class__.__name__
        finally:
            self._ready.set()
        if self.error is None and self.on_ready is not None:
            self.on_ready(self.pipeline)

    @property
    def status(self) -> str:
        """``loading``, ``ready`` or ``failed``."""
        if not self._ready.is_set():
            return "loading"
        return "failed" if self.error is not None else "ready"
