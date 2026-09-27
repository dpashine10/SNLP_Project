
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_PATH = PROJECT_ROOT / "Data" / "processed" / "researcher_profiles.csv"
EVAL_PATH = PROJECT_ROOT / "Data" / "processed" / "calculated" / "evaluation_candidates_labeled.csv"
OUT_DIR = PROJECT_ROOT / "Data" / "processed" / "calculated"
MODEL_NAME = "all-MiniLM-L6-v2"


def precision_at_k(labels: Iterable[int], k: int) -> float:
    labels = list(labels)[:k]
    if not labels:
        return 0.0
    return float(sum(labels) / k)


def recall_at_k(labels: Iterable[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return float(sum(list(labels)[:k]) / total_relevant)


def reciprocal_rank(labels: Iterable[int]) -> float:
    for rank, rel in enumerate(labels, start=1):
        if int(rel) == 1:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(labels: Iterable[int], k: int) -> float:
    labels = np.asarray(list(labels)[:k], dtype=float)
    if labels.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, labels.size + 2))
    dcg = float(np.sum((2**labels - 1) / discounts))

    ideal = np.sort(labels)[::-1]
    idcg = float(np.sum((2**ideal - 1) / discounts))
    return dcg / idcg if idcg > 0 else 0.0


def metrics_for_ranked_labels(labels: list[int]) -> dict[str, float]:
    total_relevant = int(sum(labels))
    return {
        "Precision@5": precision_at_k(labels, 5),
        "Precision@10": precision_at_k(labels, 10),
        "Recall@5": recall_at_k(labels, total_relevant, 5),
        "Recall@10": recall_at_k(labels, total_relevant, 10),
        "MRR": reciprocal_rank(labels),
        "NDCG@10": ndcg_at_k(labels, 10),
        "Relevant@10": float(sum(labels[:10])),
        "TotalRelevantInPool": float(total_relevant),
    }


def _normalize_by_query(values: np.ndarray) -> np.ndarray:
    """Min-max normalize one query's scores; constant vectors become zeros."""
    values = np.asarray(values, dtype=float)
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def evaluate_week4(model_name: str = MODEL_NAME) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run TF-IDF, Sentence Transformer, and hybrid evaluation on fixed candidates."""
    if not PROFILES_PATH.exists():
        raise FileNotFoundError(f"Missing profiles: {PROFILES_PATH}")
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Missing labeled evaluation set: {EVAL_PATH}")

    profiles = pd.read_csv(PROFILES_PATH, keep_default_na=False)
    eval_df = pd.read_csv(EVAL_PATH, keep_default_na=False)
    required = {"query", "author_id", "relevant"}
    missing = required - set(eval_df.columns)
    if missing:
        raise ValueError(f"Evaluation file is missing columns: {sorted(missing)}")

    author_to_text = dict(zip(profiles["author_id"].astype(str), profiles["profile_text"].astype(str)))
    model = SentenceTransformer(model_name)

    per_query_rows: list[dict] = []
    detailed_rows: list[dict] = []

    # Re-fit TF-IDF on the same researcher profile corpus used by the repository baseline.
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
    )
    profile_matrix = tfidf.fit_transform(profiles["profile_text"].astype(str))
    author_to_row = {str(aid): i for i, aid in enumerate(profiles["author_id"])}

    for query in eval_df["query"].drop_duplicates():
        subset = eval_df[eval_df["query"] == query].copy()
        author_ids = subset["author_id"].astype(str).tolist()
        candidate_texts = [author_to_text.get(aid, "") for aid in author_ids]

        # Baseline score recomputed from the exact corpus/profile text.
        q_tfidf = tfidf.transform([str(query)])
        candidate_rows = [author_to_row.get(aid, None) for aid in author_ids]
        valid_positions = [i for i, row in enumerate(candidate_rows) if row is not None]
        tfidf_scores = np.full(len(author_ids), -np.inf, dtype=float)
        if valid_positions:
            mat = profile_matrix[[candidate_rows[i] for i in valid_positions]]
            sims = cosine_similarity(q_tfidf, mat).ravel()
            for pos, score in zip(valid_positions, sims):
                tfidf_scores[pos] = float(score)

        q_emb = model.encode(str(query), normalize_embeddings=True, show_progress_bar=False)
        c_emb = model.encode(candidate_texts, normalize_embeddings=True, show_progress_bar=False)
        semantic_scores = np.dot(c_emb, q_emb).astype(float)
        hybrid_scores = 0.7 * _normalize_by_query(semantic_scores) + 0.3 * _normalize_by_query(np.nan_to_num(tfidf_scores, nan=0.0, neginf=0.0))

        subset["tfidf_score_week4"] = tfidf_scores
        subset["semantic_score_week4"] = semantic_scores
        subset["hybrid_score_week4"] = hybrid_scores

        score_map = {
            "TF-IDF Baseline": "tfidf_score_week4",
            "Sentence Transformer": "semantic_score_week4",
            "Hybrid": "hybrid_score_week4",
        }

        for model_label, score_col in score_map.items():
            ranked = subset.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
            labels = ranked["relevant"].astype(int).tolist()
            m = metrics_for_ranked_labels(labels)
            per_query_rows.append({"query": query, "Model": model_label, **m})
            for rank, row in ranked.iterrows():
                detailed_rows.append(
                    {
                        "query": query,
                        "Model": model_label,
                        "rank": rank + 1,
                        "author_id": row["author_id"],
                        "author_name": row.get("author_name", ""),
                        "relevant": int(row["relevant"]),
                        "score": float(row[score_col]),
                    }
                )

    per_query = pd.DataFrame(per_query_rows)
    comparison = (
        per_query.groupby("Model", as_index=False)[
            ["Precision@5", "Precision@10", "Recall@5", "Recall@10", "MRR", "NDCG@10", "Relevant@10"]
        ]
        .mean()
        .sort_values("Model")
    )
    detailed = pd.DataFrame(detailed_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(OUT_DIR / "week4_query_metrics.csv", index=False)
    comparison.to_csv(OUT_DIR / "week4_model_comparison.csv", index=False)
    detailed.to_csv(OUT_DIR / "week4_rankings.csv", index=False)

    # Identify cases where the top-5 membership changes between baseline and improved.
    changes: list[dict] = []
    for query in eval_df["query"].drop_duplicates():
        q = detailed[detailed["query"] == query]
        tf5 = set(q[q["Model"] == "TF-IDF Baseline"].nsmallest(5, "rank")["author_id"])
        sb5 = set(q[q["Model"] == "Sentence Transformer"].nsmallest(5, "rank")["author_id"])
        for aid in sorted(tf5.symmetric_difference(sb5)):
            row = q[q["author_id"] == aid].iloc[0]
            changes.append(
                {
                    "query": query,
                    "author_id": aid,
                    "author_name": row["author_name"],
                    "relevant": row["relevant"],
                    "tfidf_rank": int(q[(q["Model"] == "TF-IDF Baseline") & (q["author_id"] == aid)]["rank"].iloc[0]),
                    "semantic_rank": int(q[(q["Model"] == "Sentence Transformer") & (q["author_id"] == aid)]["rank"].iloc[0]),
                }
            )
    pd.DataFrame(changes).to_csv(OUT_DIR / "week4_error_cases.csv", index=False)

    return comparison, per_query, detailed


if __name__ == "__main__":
    comp, _, _ = evaluate_week4()
    print("\nWeek 4 model comparison")
    print(comp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
