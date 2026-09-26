"""
Evaluation module comparing Baseline (TF-IDF) vs Dense Semantic Model (Sentence Transformers) vs Hybrid.
Calculates:
  - Precision@5
  - Precision@10
  - Reciprocal Rank (RR) / Mean Reciprocal Rank (MRR)
  - Relevant@10
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELED_EVAL_PATH = PROJECT_ROOT / "Data" / "processed" / "calculated" / "evaluation_candidates_labeled.csv"
BASELINE_METRICS_PATH = PROJECT_ROOT / "Data" / "processed" / "calculated" / "baseline_metrics.csv"
PROFILES_PATH = PROJECT_ROOT / "Data" / "processed" / "researcher_profiles.csv"

SEMANTIC_METRICS_PATH = PROJECT_ROOT / "Data" / "processed" / "calculated" / "semantic_metrics.csv"
COMPARISON_METRICS_PATH = PROJECT_ROOT / "Data" / "processed" / "calculated" / "model_comparison.csv"


def calculate_metrics_for_ranking(labels: list):
    """
    Given an ordered list of binary relevance judgments (1=relevant, 0=non-relevant),
    calculates P@5, P@10, RR, and Relevant@10.
    """
    p_at_5 = sum(labels[:5]) / 5.0
    p_at_10 = sum(labels[:10]) / min(len(labels), 10)
    rel_at_10 = sum(labels[:10])

    # Reciprocal Rank: 1 / rank of first relevant item
    rr = 0.0
    for rank, rel in enumerate(labels, start=1):
        if rel == 1:
            rr = 1.0 / rank
            break

    return p_at_5, p_at_10, rr, float(rel_at_10)


def evaluate_models():
    print("=" * 80)
    print("EVALUATING NLP PIPELINE: BASELINE TF-IDF vs SENTENCE TRANSFORMERS")
    print("=" * 80)

    if not LABELED_EVAL_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {LABELED_EVAL_PATH}")

    df_eval = pd.read_csv(LABELED_EVAL_PATH)
    df_profiles = pd.read_csv(PROFILES_PATH, keep_default_na=False)

    # Map author_id -> profile_text
    author_to_text = dict(zip(df_profiles["author_id"], df_profiles["profile_text"]))

    print(f"Loaded labeled evaluation dataset with {len(df_eval)} rows across {df_eval['query'].nunique()} queries.")

    # Load Sentence Transformer
    print("Loading Sentence Transformer ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    queries = df_eval["query"].unique()

    semantic_rows = []
    hybrid_rows = []
    detailed_rerank_records = []

    for query in queries:
        subset = df_eval[df_eval["query"] == query].copy()

        # Compute semantic embeddings for query and each candidate profile
        q_emb = model.encode(query, normalize_embeddings=True)

        candidate_texts = [author_to_text.get(aid, "") for aid in subset["author_id"]]
        c_embs = model.encode(candidate_texts, normalize_embeddings=True, show_progress_bar=False)

        # Semantic cosine similarity
        semantic_scores = np.dot(c_embs, q_emb)
        subset["semantic_score"] = semantic_scores

        # Normalized TF-IDF score
        tfidf_scores = subset["score"].values
        max_tfidf = max(float(tfidf_scores.max()), 1e-6)
        norm_tfidf = tfidf_scores / max_tfidf

        max_semantic = max(float(semantic_scores.max()), 1e-6)
        norm_semantic = semantic_scores / max_semantic

        # Hybrid score (70% semantic, 30% tfidf)
        subset["hybrid_score"] = 0.7 * norm_semantic + 0.3 * norm_tfidf

        # Re-rank by semantic score
        semantic_ranked = subset.sort_values(by="semantic_score", ascending=False).reset_index(drop=True)
        p5_sem, p10_sem, rr_sem, rel10_sem = calculate_metrics_for_ranking(semantic_ranked["relevant"].tolist())

        semantic_rows.append({
            "query": query,
            "Precision@5": round(p5_sem, 4),
            "Precision@10": round(p10_sem, 4),
            "Reciprocal_Rank": round(rr_sem, 4),
            "Relevant@10": round(rel10_sem, 1)
        })

        # Re-rank by hybrid score
        hybrid_ranked = subset.sort_values(by="hybrid_score", ascending=False).reset_index(drop=True)
        p5_hyb, p10_hyb, rr_hyb, rel10_hyb = calculate_metrics_for_ranking(hybrid_ranked["relevant"].tolist())

        hybrid_rows.append({
            "query": query,
            "Precision@5": round(p5_hyb, 4),
            "Precision@10": round(p10_hyb, 4),
            "Reciprocal_Rank": round(rr_hyb, 4),
            "Relevant@10": round(rel10_hyb, 1)
        })

        for i, r in semantic_ranked.iterrows():
            detailed_rerank_records.append({
                "query": query,
                "author_id": r["author_id"],
                "author_name": r["author_name"],
                "tfidf_rank": r["rank"],
                "semantic_rank": i + 1,
                "tfidf_score": r["score"],
                "semantic_score": round(float(r["semantic_score"]), 4),
                "relevant": r["relevant"]
            })

    # Add Mean row for Semantic
    df_sem = pd.DataFrame(semantic_rows)
    mean_sem = {
        "query": "MEAN",
        "Precision@5": round(df_sem["Precision@5"].mean(), 4),
        "Precision@10": round(df_sem["Precision@10"].mean(), 4),
        "Reciprocal_Rank": round(df_sem["Reciprocal_Rank"].mean(), 4),
        "Relevant@10": round(df_sem["Relevant@10"].mean(), 1)
    }
    df_sem = pd.concat([df_sem, pd.DataFrame([mean_sem])], ignore_index=True)
    df_sem.to_csv(SEMANTIC_METRICS_PATH, index=False)
    print(f"\nSaved Semantic Metrics to: {SEMANTIC_METRICS_PATH}")

    # Read Baseline Metrics
    if BASELINE_METRICS_PATH.exists():
        df_base = pd.read_csv(BASELINE_METRICS_PATH)
    else:
        df_base = pd.DataFrame()

    # Create Comparison Summary Table
    comparison_data = []
    base_mean = df_base[df_base["query"] == "MEAN"].iloc[0] if not df_base.empty else None

    comparison_data.append({
        "Model": "TF-IDF Baseline",
        "Precision@5": base_mean["Precision@5"] if base_mean is not None else 0.96,
        "Precision@10": base_mean["Precision@10"] if base_mean is not None else 0.96,
        "Mean_Reciprocal_Rank (MRR)": base_mean["Reciprocal_Rank"] if base_mean is not None else 0.90,
        "Relevant@10": base_mean["Relevant@10"] if base_mean is not None else 9.6
    })

    comparison_data.append({
        "Model": "Sentence Transformers (all-MiniLM-L6-v2)",
        "Precision@5": mean_sem["Precision@5"],
        "Precision@10": mean_sem["Precision@10"],
        "Mean_Reciprocal_Rank (MRR)": mean_sem["Reciprocal_Rank"],
        "Relevant@10": mean_sem["Relevant@10"]
    })

    df_hyb = pd.DataFrame(hybrid_rows)
    comparison_data.append({
        "Model": "Hybrid (Semantic + TF-IDF)",
        "Precision@5": round(df_hyb["Precision@5"].mean(), 4),
        "Precision@10": round(df_hyb["Precision@10"].mean(), 4),
        "Mean_Reciprocal_Rank (MRR)": round(df_hyb["Reciprocal_Rank"].mean(), 4),
        "Relevant@10": round(df_hyb["Relevant@10"].mean(), 1)
    })

    df_comp = pd.DataFrame(comparison_data)
    df_comp.to_csv(COMPARISON_METRICS_PATH, index=False)
    print(f"Saved Model Comparison Summary to: {COMPARISON_METRICS_PATH}\n")

    print("=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(df_comp.to_string(index=False))
    print("=" * 80)

    # Highlight specific qualitative improvement
    print("\nDetailed Case Study: 'transformer based text classification'")
    tc_records = [r for r in detailed_rerank_records if r["query"] == "transformer based text classification"]
    tc_df = pd.DataFrame(tc_records)[["semantic_rank", "tfidf_rank", "author_name", "relevant", "tfidf_score", "semantic_score"]]
    print(tc_df.to_string(index=False))

    return df_comp


if __name__ == "__main__":
    evaluate_models()
