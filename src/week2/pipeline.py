"""
Core NLP Pipeline for Research Discovery & Collaboration Engine (Version 1).
Supports:
  1. Dense Semantic Similarity using Sentence Transformers (all-MiniLM-L6-v2).
  2. Sparse Baseline using TF-IDF (Unigram + Bigram).
  3. Hybrid Retrieval (Dense Semantic + Sparse Keyword).
  4. Contextual Attribution: Identifies the most relevant paper and related topics.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.week6.filters import is_non_research_artifact

for _optional_ext in ("torchvision", "torchaudio"):
    if _optional_ext not in sys.modules:
        try:
            __import__(_optional_ext)
        except (OSError, RuntimeError, ImportError):
            sys.modules[_optional_ext] = None  # type: ignore[assignment]

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPERS_PATH = PROJECT_ROOT / "Data" / "processed" / "papers.csv"
AUTHORS_PATH = PROJECT_ROOT / "Data" / "processed" / "paper_author.csv"
PROFILES_PATH = PROJECT_ROOT / "Data" / "processed" / "researcher_profiles.csv"

PAPER_EMB_PATH = PROJECT_ROOT / "Data" / "processed" / "paper_embeddings.npz"
RESEARCHER_EMB_PATH = PROJECT_ROOT / "Data" / "processed" / "researcher_embeddings.npz"

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TOPIC_BADGES = 6
_IGNORED_TOPIC_TOKENS = frozenset({"", "nan", "none", "n/a", "unknown"})


def _format_year(year_val: Any) -> str:
    if pd.notna(year_val) and str(year_val).strip() != "":
        try:
            return str(int(float(year_val)))
        except ValueError:
            return str(year_val).strip()
    return "Unknown"


def _build_paper_info(
    paper: Mapping[str, Any],
    score: float,
    *,
    matched_papers: Optional[int] = None,
    discounted_score: Optional[float] = None,
) -> Dict[str, Any]:
    authors_raw = str(paper.get("authors", ""))
    authors_count = (
        max(1, len([a for a in authors_raw.split(";") if a.strip()]))
        if authors_raw.strip()
        else 1
    )
    info: Dict[str, Any] = {
        "title": paper.get("title", "Unknown Title"),
        "year": _format_year(paper.get("publication_year", "")),
        "work_id": str(paper.get("work_id", "")).strip(),
        "doi": str(paper.get("doi", "")).strip(),
        "abstract": str(paper.get("abstract", "")).strip(),
        "institutions": str(paper.get("institutions", "")).strip(),
        "authors_count": authors_count,
        "paper_score": round(float(score), 4),
    }
    if matched_papers is not None:
        info["matched_papers"] = matched_papers
    if discounted_score is not None:
        info["discounted_score"] = round(float(discounted_score), 4)
    return info


def _extract_clean_topics(
    evidence_papers: Sequence[Mapping[str, Any]],
    fallback_papers: Sequence[Mapping[str, Any]] = (),
    limit: int = MAX_TOPIC_BADGES,
) -> List[str]:
    """Split on '|' and ',', deduplicate, prioritize matched evidence papers, and cap at ``limit``."""
    topics: List[str] = []
    seen: set[str] = set()

    def _collect(papers: Sequence[Mapping[str, Any]], field_name: str) -> None:
        for paper in papers:
            if len(topics) >= limit:
                return
            raw = str(paper.get(field_name, ""))
            for seg in re.split(r"[|,]", raw):
                clean = " ".join(seg.strip().split())
                lowered = clean.lower()
                if lowered not in _IGNORED_TOPIC_TOKENS and lowered not in seen:
                    seen.add(lowered)
                    topics.append(clean)
                    if len(topics) >= limit:
                        return

    _collect(evidence_papers, "topics")
    _collect(evidence_papers, "keywords")
    if not topics and fallback_papers:
        _collect(fallback_papers, "topics")
        _collect(fallback_papers, "keywords")
    return topics[:limit]


class ResearchDiscoveryPipeline:
    def __init__(self, model_name: str = MODEL_NAME, auto_load: bool = True):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        self.profiles_df: Optional[pd.DataFrame] = None
        self.papers_df: Optional[pd.DataFrame] = None
        self.papers_by_id: Dict[str, Dict[str, Any]] = {}

        self.paper_embeddings: Optional[np.ndarray] = None
        self._norm_paper_embeddings: Optional[np.ndarray] = None
        self.paper_work_ids: Optional[np.ndarray] = None
        self.work_to_emb_idx: Dict[str, int] = {}

        self.researcher_embeddings: Optional[np.ndarray] = None
        self.researcher_author_ids: Optional[np.ndarray] = None
        self.author_to_emb_idx: Dict[str, int] = {}
        self.profile_by_author_id: Dict[str, Dict[str, Any]] = {}
        self._valid_paper_mask: Optional[np.ndarray] = None
        self._paper_authors_by_emb_idx: List[Tuple[Tuple[str, str], ...]] = []
        self._valid_profile_mask: Optional[np.ndarray] = None
        self._paper_index_built: bool = False

        if auto_load:
            self.load()

    def load(self):
        """Loads data, initializes models, and loads precomputed embeddings."""
        print("Loading researcher profiles...")
        self.profiles_df = pd.read_csv(PROFILES_PATH, keep_default_na=False)

        print("Loading papers dataset...")
        self.papers_df = pd.read_csv(PAPERS_PATH, keep_default_na=False)
        self.papers_by_id = {
            row["work_id"]: row.to_dict() for _, row in self.papers_df.iterrows()
        }

        print(f"Loading Sentence Transformer model ({self.model_name})...")
        self.model = SentenceTransformer(self.model_name)

        print("Fitting TF-IDF baseline vectorizer...")
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.profiles_df["profile_text"])

        # Load dense embeddings
        if not PAPER_EMB_PATH.exists() or not RESEARCHER_EMB_PATH.exists():
            print("Precomputed embeddings not found. Generating now...")
            from src.week3.build_embeddings import build_embeddings
            build_embeddings()

        print("Loading precomputed embeddings from cache...")
        paper_data = np.load(PAPER_EMB_PATH)
        self.paper_work_ids = paper_data["work_ids"]
        self.paper_embeddings = paper_data["embeddings"]
        self.work_to_emb_idx = {wid: i for i, wid in enumerate(self.paper_work_ids)}

        researcher_data = np.load(RESEARCHER_EMB_PATH)
        self.researcher_author_ids = researcher_data["author_ids"]
        self.researcher_embeddings = researcher_data["embeddings"]
        self.author_to_emb_idx = {aid: i for i, aid in enumerate(self.researcher_author_ids)}
        self._ensure_paper_author_index()

        print("Pipeline initialization complete!")

    def _ensure_paper_author_index(self) -> None:
        """Build paper-level author lookup and non-research artifact masks once."""
        if self._paper_index_built:
            return
        if self.profiles_df is not None and not self.profile_by_author_id:
            self.profile_by_author_id = {
                str(row["author_id"]): row.to_dict() for _, row in self.profiles_df.iterrows()
            }
            valid_profiles = np.ones(len(self.profiles_df), dtype=bool)
            for idx, row in enumerate(self.profiles_df.itertuples(index=False)):
                pids = [pid for pid in str(getattr(row, "paper_ids", "")).split("|") if pid]
                if pids and self.papers_by_id:
                    has_real_paper = any(
                        pid in self.papers_by_id
                        and not is_non_research_artifact(self.papers_by_id[pid].get("title"))
                        for pid in pids
                    )
                    if not has_real_paper:
                        valid_profiles[idx] = False
            self._valid_profile_mask = valid_profiles

        if self.paper_work_ids is not None:
            n_papers = len(self.paper_work_ids)
            valid_mask = np.zeros(n_papers, dtype=bool)
            authors_by_idx: List[Tuple[Tuple[str, str], ...]] = [() for _ in range(n_papers)]
            for i, wid_raw in enumerate(self.paper_work_ids):
                wid = str(wid_raw)
                paper = self.papers_by_id.get(wid)
                if paper is None or is_non_research_artifact(paper.get("title")):
                    continue
                names = [n.strip() for n in str(paper.get("authors", "")).split(";") if n.strip()]
                aids = [a.strip() for a in str(paper.get("author_ids", "")).split("|") if a.strip()]
                if not names or not aids:
                    continue
                pairs = tuple(
                    (aids[j] if j < len(aids) else f"auth_{name}", name)
                    for j, name in enumerate(names)
                )
                authors_by_idx[i] = pairs
                valid_mask[i] = True
            self._valid_paper_mask = valid_mask
            self._paper_authors_by_emb_idx = authors_by_idx

        if self.paper_embeddings is not None:
            emb = np.asarray(self.paper_embeddings, dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._norm_paper_embeddings = emb / norms
        self._paper_index_built = True

    def _get_query_embedding(self, query: str) -> np.ndarray:
        q_emb = self.model.encode(query, normalize_embeddings=True)
        return np.asarray(q_emb, dtype=np.float32)

    def _get_query_tfidf(self, query: str):
        return self.vectorizer.transform([query])

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "semantic",
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Rank researchers matching the query.

        Args:
            query: Research topic, problem description, or paper abstract.
            top_k: Number of top researchers to return.
            mode: 'paper_discount', 'semantic', 'tfidf', or 'hybrid'.
            alpha: Weight for dense semantic similarity in hybrid mode (0.0 to 1.0).

        Returns:
            List of structured researcher result dictionaries.
        """
        if not query or not query.strip():
            return []

        query = query.strip()
        self._ensure_paper_author_index()

        if mode == "paper_discount":
            return self._search_paper_discount(query=query, top_k=top_k)

        query_emb: Optional[np.ndarray] = None
        if mode == "semantic":
            query_emb = self._get_query_embedding(query)
            # Dot product is cosine similarity because embeddings are L2 normalized
            scores = np.dot(self.researcher_embeddings, query_emb)

        elif mode == "tfidf":
            query_tfidf = self._get_query_tfidf(query)
            scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()

        elif mode == "hybrid":
            query_emb = self._get_query_embedding(query)
            dense_scores = np.dot(self.researcher_embeddings, query_emb)

            query_tfidf = self._get_query_tfidf(query)
            sparse_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()

            # Normalize scores to [0, 1] range for fair blending if max > 0
            d_max = max(float(dense_scores.max()), 1e-6)
            s_max = max(float(sparse_scores.max()), 1e-6)
            norm_dense = np.clip(dense_scores / d_max, 0.0, 1.0)
            norm_sparse = np.clip(sparse_scores / s_max, 0.0, 1.0)

            scores = (alpha * norm_dense) + ((1.0 - alpha) * norm_sparse)

        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Use 'paper_discount', 'semantic', 'tfidf', or 'hybrid'."
            )

        ranked_indices = np.argsort(scores)[::-1]
        if self._valid_profile_mask is not None and len(self._valid_profile_mask) == len(scores):
            top_indices = [int(i) for i in ranked_indices if self._valid_profile_mask[i]][:top_k]
        else:
            top_indices = [int(i) for i in ranked_indices[:top_k]]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            row = self.profiles_df.iloc[idx]
            author_id = row["author_id"]
            author_name = row["author_name"]
            n_papers = int(row["n_papers"])
            score = float(scores[idx])
            raw_inst = str(row["institution_name"]).strip() if "institution_name" in row else ""

            # Find researcher's most relevant paper(s)
            paper_ids = [pid for pid in str(row["paper_ids"]).split("|") if pid]
            relevant_paper, related_topics = self._find_relevant_paper_and_topics(
                paper_ids=paper_ids,
                query=query,
                query_emb=query_emb
            )
            institution_name = raw_inst or str(relevant_paper.get("institutions", "")).strip()
            matching_papers = relevant_paper.get("top_papers") or [relevant_paper]

            results.append({
                "rank": rank,
                "author_id": author_id,
                "author_name": author_name,
                "institution_name": institution_name,
                "similarity_score": round(score, 4),
                "n_papers": n_papers,
                "matched_papers": int(relevant_paper.get("matched_papers", 1)),
                "relevant_paper": relevant_paper,
                "matching_papers": matching_papers,
                "related_topics": related_topics,
                "mode": mode
            })

        return results

    def _search_paper_discount(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Paper-level dense retrieval with top-3 mean and sqrt(co-author) discounting (Week 5/6)."""
        if (
            self._norm_paper_embeddings is None
            or self.paper_work_ids is None
            or self._valid_paper_mask is None
            or not np.any(self._valid_paper_mask)
        ):
            return []

        query_emb = self._get_query_embedding(query)
        norm = float(np.linalg.norm(query_emb))
        if norm > 0:
            query_emb = query_emb / norm

        paper_sims = np.dot(self._norm_paper_embeddings, query_emb)
        masked_sims = np.where(self._valid_paper_mask, paper_sims, -2.0)
        valid_count = int(np.count_nonzero(self._valid_paper_mask))
        paper_top_k = min(max(100, top_k * 10), valid_count)

        if paper_top_k < len(masked_sims):
            cand_idx = np.argpartition(masked_sims, -paper_top_k)[-paper_top_k:]
        else:
            cand_idx = np.arange(len(masked_sims))
        cand_idx = cand_idx[self._valid_paper_mask[cand_idx]]
        sorted_paper_idx = sorted(
            (int(i) for i in cand_idx),
            key=lambda i: (-float(paper_sims[i]), str(self.paper_work_ids[i])),
        )[:paper_top_k]

        # Group hits by author_id, applying 1 / sqrt(n_authors) per paper hit
        author_hits: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
        author_names: Dict[str, str] = {}
        for emb_idx in sorted_paper_idx:
            raw_sim = float(paper_sims[emb_idx])
            wid = str(self.paper_work_ids[emb_idx])
            authors = self._paper_authors_by_emb_idx[emb_idx]
            discount = math.sqrt(max(len(authors), 1))
            disc_sim = raw_sim / discount
            for author_id, author_name in authors:
                if author_id not in author_names:
                    author_names[author_id] = author_name
                author_hits[author_id].append((disc_sim, raw_sim, wid))

        scored_authors: List[Tuple[float, str]] = []
        for author_id, hits in author_hits.items():
            top3_discounted = sorted((h[0] for h in hits), reverse=True)[:3]
            mean_score = sum(top3_discounted) / len(top3_discounted)
            scored_authors.append((mean_score, author_id))

        scored_authors.sort(key=lambda item: (-item[0], item[1]))

        results: List[Dict[str, Any]] = []
        for rank, (author_score, author_id) in enumerate(scored_authors[:top_k], start=1):
            hits = author_hits[author_id]
            top_hits = hits[:3]
            best_disc_sim, best_raw_sim, best_wid = top_hits[0]
            best_paper = self.papers_by_id.get(best_wid, {})

            profile = self.profile_by_author_id.get(author_id)
            if profile is not None:
                author_name = str(profile.get("author_name") or author_names[author_id])
                raw_n_papers = str(profile.get("n_papers", "")).strip()
                n_papers = int(float(raw_n_papers)) if raw_n_papers else len(hits)
                institution_name = str(profile.get("institution_name", "")).strip()
            else:
                author_name = author_names[author_id]
                n_papers = len(hits)
                institution_name = ""
            if not institution_name:
                institution_name = str(best_paper.get("institutions", "")).strip()

            matching_papers = [
                _build_paper_info(
                    self.papers_by_id.get(wid, {"work_id": wid}),
                    raw_sim,
                    matched_papers=len(hits) if idx == 0 else None,
                    discounted_score=disc_sim,
                )
                for idx, (disc_sim, raw_sim, wid) in enumerate(top_hits)
            ]
            rel_paper_info = matching_papers[0]
            evidence_dicts = [self.papers_by_id.get(wid, {}) for _, _, wid in top_hits]
            topics_list = _extract_clean_topics(evidence_dicts, limit=MAX_TOPIC_BADGES)

            results.append({
                "rank": rank,
                "author_id": author_id,
                "author_name": author_name,
                "institution_name": institution_name,
                "similarity_score": round(author_score, 4),
                "n_papers": n_papers,
                "matched_papers": len(hits),
                "relevant_paper": rel_paper_info,
                "matching_papers": matching_papers,
                "related_topics": topics_list,
                "mode": "paper_discount",
            })

        return results

    def _find_relevant_paper_and_topics(
        self,
        paper_ids: List[str],
        query: str,
        query_emb: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Finds the researcher's top matching papers and extracts clean topics from matched evidence."""
        if not paper_ids:
            return {"title": "N/A", "year": "N/A", "score": 0.0}, []

        candidate_papers = [self.papers_by_id[pid] for pid in paper_ids if pid in self.papers_by_id]
        if not candidate_papers:
            return {"title": "N/A", "year": "N/A", "score": 0.0}, []

        non_artifact_papers = [
            p for p in candidate_papers if not is_non_research_artifact(p.get("title"))
        ]
        if non_artifact_papers:
            candidate_papers = non_artifact_papers

        ranked_pairs: List[Tuple[Dict[str, Any], float]] = [(candidate_papers[0], 0.0)]
        matched_papers = 1

        if query_emb is not None and self.paper_embeddings is not None:
            # Score each paper using dense cosine similarity
            indexed_papers = [
                p for p in candidate_papers if p["work_id"] in self.work_to_emb_idx
            ]
            if indexed_papers:
                paper_indices = [self.work_to_emb_idx[p["work_id"]] for p in indexed_papers]
                emb_source = (
                    self._norm_paper_embeddings
                    if self._norm_paper_embeddings is not None
                    else self.paper_embeddings
                )
                paper_embs = emb_source[paper_indices]
                paper_scores = np.dot(paper_embs, query_emb)
                order = sorted(
                    range(len(indexed_papers)),
                    key=lambda i: (-float(paper_scores[i]), str(indexed_papers[i].get("work_id", ""))),
                )
                ranked_pairs = [(indexed_papers[i], float(paper_scores[i])) for i in order]
                matched_papers = max(1, int(np.sum(paper_scores >= 0.30)))
        else:
            # Fallback to TF-IDF score
            paper_texts = [
                f"{p.get('title', '')} {p.get('abstract', '')}" for p in candidate_papers
            ]
            p_mat = self.vectorizer.transform(paper_texts)
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, p_mat).flatten()
            order = sorted(
                range(len(candidate_papers)),
                key=lambda i: (-float(sims[i]), str(candidate_papers[i].get("work_id", ""))),
            )
            ranked_pairs = [(candidate_papers[i], float(sims[i])) for i in order]
            matched_papers = max(1, int(np.sum(sims >= 0.10)))

        top_pairs = ranked_pairs[:3]
        matching_papers = [
            _build_paper_info(paper, score, matched_papers=matched_papers if idx == 0 else None)
            for idx, (paper, score) in enumerate(top_pairs)
        ]
        evidence_papers = [paper for paper, _ in top_pairs]
        topics_list = _extract_clean_topics(
            evidence_papers,
            fallback_papers=candidate_papers,
            limit=MAX_TOPIC_BADGES,
        )

        rel_paper_info = dict(matching_papers[0])
        rel_paper_info["top_papers"] = matching_papers

        return rel_paper_info, topics_list
