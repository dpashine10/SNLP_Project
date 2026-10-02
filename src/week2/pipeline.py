"""
Core NLP Pipeline for Research Discovery & Collaboration Engine (Version 1).
Supports:
  1. Dense Semantic Similarity using Sentence Transformers (all-MiniLM-L6-v2).
  2. Sparse Baseline using TF-IDF (Unigram + Bigram).
  3. Hybrid Retrieval (Dense Semantic + Sparse Keyword).
  4. Contextual Attribution: Identifies the most relevant paper and related topics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPERS_PATH = PROJECT_ROOT / "Data" / "processed" / "papers.csv"
AUTHORS_PATH = PROJECT_ROOT / "Data" / "processed" / "paper_author.csv"
PROFILES_PATH = PROJECT_ROOT / "Data" / "processed" / "researcher_profiles.csv"

PAPER_EMB_PATH = PROJECT_ROOT / "Data" / "processed" / "paper_embeddings.npz"
RESEARCHER_EMB_PATH = PROJECT_ROOT / "Data" / "processed" / "researcher_embeddings.npz"

MODEL_NAME = "all-MiniLM-L6-v2"


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
        self.paper_work_ids: Optional[np.ndarray] = None
        self.work_to_emb_idx: Dict[str, int] = {}

        self.researcher_embeddings: Optional[np.ndarray] = None
        self.researcher_author_ids: Optional[np.ndarray] = None
        self.author_to_emb_idx: Dict[str, int] = {}

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

        print("Pipeline initialization complete!")

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
            mode: 'semantic' (Sentence Transformers), 'tfidf' (Baseline), or 'hybrid'.
            alpha: Weight for dense semantic similarity in hybrid mode (0.0 to 1.0).

        Returns:
            List of structured researcher result dictionaries.
        """
        if not query or not query.strip():
            return []

        query = query.strip()
        num_profiles = len(self.profiles_df)

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
            raise ValueError(f"Unknown mode '{mode}'. Use 'semantic', 'tfidf', or 'hybrid'.")

        top_indices = np.argsort(scores)[::-1][:top_k]

        # Contextual attribution for each top researcher
        query_emb = self._get_query_embedding(query) if mode != "tfidf" else None

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            row = self.profiles_df.iloc[idx]
            author_id = row["author_id"]
            author_name = row["author_name"]
            n_papers = int(row["n_papers"])
            score = float(scores[idx])

            # Find researcher's most relevant paper
            paper_ids = [pid for pid in str(row["paper_ids"]).split("|") if pid]
            relevant_paper, related_topics = self._find_relevant_paper_and_topics(
                paper_ids=paper_ids,
                query=query,
                query_emb=query_emb
            )

            results.append({
                "rank": rank,
                "author_id": author_id,
                "author_name": author_name,
                "similarity_score": round(score, 4),
                "n_papers": n_papers,
                "relevant_paper": relevant_paper,
                "related_topics": related_topics,
                "mode": mode
            })

        return results

    def _find_relevant_paper_and_topics(
        self,
        paper_ids: List[str],
        query: str,
        query_emb: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Finds the researcher's paper closest to the query and extracts salient topics."""
        if not paper_ids:
            return {"title": "N/A", "year": "N/A", "score": 0.0}, []

        candidate_papers = [self.papers_by_id[pid] for pid in paper_ids if pid in self.papers_by_id]
        if not candidate_papers:
            return {"title": "N/A", "year": "N/A", "score": 0.0}, []

        best_paper = candidate_papers[0]
        best_score = 0.0

        if query_emb is not None and self.paper_embeddings is not None:
            # Score each paper using dense cosine similarity
            paper_indices = [
                self.work_to_emb_idx[p["work_id"]]
                for p in candidate_papers
                if p["work_id"] in self.work_to_emb_idx
            ]
            if paper_indices:
                paper_embs = self.paper_embeddings[paper_indices]
                paper_scores = np.dot(paper_embs, query_emb)
                best_sub_idx = int(np.argmax(paper_scores))
                best_score = float(paper_scores[best_sub_idx])
                best_paper = candidate_papers[best_sub_idx]
        else:
            # Fallback to TF-IDF score
            paper_texts = [
                f"{p.get('title', '')} {p.get('abstract', '')}" for p in candidate_papers
            ]
            p_mat = self.vectorizer.transform(paper_texts)
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, p_mat).flatten()
            best_sub_idx = int(np.argmax(sims))
            best_score = float(sims[best_sub_idx])
            best_paper = candidate_papers[best_sub_idx]

        # Extract topics
        topics_set = []
        for p in candidate_papers:
            for t in str(p.get("topics", "")).split("|"):
                t = t.strip()
                if t and t not in topics_set:
                    topics_set.append(t)
            for k in str(p.get("keywords", "")).split("|"):
                k = k.strip()
                if k and k not in topics_set:
                    topics_set.append(k)

        year_val = best_paper.get("publication_year", "")
        if pd.notna(year_val) and year_val != "":
            year_str = str(int(float(year_val)))
        else:
            year_str = "Unknown"

        rel_paper_info = {
            "title": best_paper.get("title", "Unknown Title"),
            "year": year_str,
            "work_id": best_paper.get("work_id", ""),
            "paper_score": round(best_score, 4)
        }

        return rel_paper_info, topics_set[:5]
