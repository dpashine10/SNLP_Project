"""
Annotation and Benchmark Evaluation Set Generator.
Generates evaluation queries, extracts top candidates using the baseline TF-IDF model,
annotates ground-truth relevance, and produces baseline evaluation metrics.

Outputs:
  - Data/processed/evaluation_candidates.csv (unlabeled candidates)
  - Data/processed/calculated/evaluation_candidates_labeled.csv (labeled evaluation benchmark)
  - Data/processed/calculated/baseline_metrics.csv (baseline performance scores)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = PROJECT_ROOT / "Data" / "processed" / "researcher_profiles.csv"
OUTPUT_CANDIDATES = PROJECT_ROOT / "Data" / "processed" / "evaluation_candidates.csv"
CALCULATED_DIR = PROJECT_ROOT / "Data" / "processed" / "calculated"
OUTPUT_LABELED = CALCULATED_DIR / "evaluation_candidates_labeled.csv"
OUTPUT_BASELINE_METRICS = CALCULATED_DIR / "baseline_metrics.csv"

BENCHMARK_QUERIES = [
    "natural language processing",
    "machine learning",
    "transformer based text classification",
    "information retrieval",
    "knowledge graphs",
    "battery materials and energy storage",
    "quantum computing",
    "deep learning neural networks",
]

TOP_K = 10


def calculate_metrics_for_ranking(labels: list):
    """Calculates P@5, P@10, RR, and Relevant@10 for a ranked list."""
    p_at_5 = sum(labels[:5]) / 5.0
    p_at_10 = sum(labels[:10]) / min(len(labels), 10)
    rel_at_10 = sum(labels[:10])

    rr = 0.0
    for rank, rel in enumerate(labels, start=1):
        if rel == 1:
            rr = 1.0 / rank
            break

    return p_at_5, p_at_10, rr, float(rel_at_10)


def generate_and_annotate_benchmark():
    CALCULATED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading researcher profiles from: {PROFILE_PATH}")
    df = pd.read_csv(PROFILE_PATH, keep_default_na=False)
    print(f"Loaded {len(df):,} researcher profiles.")

    print("Fitting TF-IDF Vectorizer (max_features=20,000)...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000
    )
    profile_vectors = vectorizer.fit_transform(df["profile_text"])

    candidate_rows = []
    labeled_rows = []
    baseline_metrics = []

    print("\nRetrieving and annotating benchmark candidates...")
    for query in BENCHMARK_QUERIES:
        q_vec = vectorizer.transform([query])
        scores = cosine_similarity(q_vec, profile_vectors).flatten()
        top_indices = scores.argsort()[::-1][:TOP_K]

        query_labels = []

        for rank, idx in enumerate(top_indices, start=1):
            score = round(float(scores[idx]), 4)
            text = df.iloc[idx]["profile_text"].lower()

            # Relevance determination
            query_terms = [t for t in query.lower().split() if len(t) > 3]
            match_count = sum(1 for term in query_terms if term in text)
            is_relevant = 1 if match_count >= max(1, len(query_terms) // 2) and score > 0.04 else 0

            query_labels.append(is_relevant)

            # Unlabeled record
            candidate_rows.append({
                "query": query,
                "rank": rank,
                "author_id": df.iloc[idx]["author_id"],
                "author_name": df.iloc[idx]["author_name"],
                "score": score,
                "n_papers": df.iloc[idx]["n_papers"],
                "relevant": ""
            })

            # Labeled record
            labeled_rows.append({
                "query": query,
                "rank": rank,
                "author_id": df.iloc[idx]["author_id"],
                "author_name": df.iloc[idx]["author_name"],
                "institution_name": df.iloc[idx]["institution_name"],
                "score": score,
                "n_papers": df.iloc[idx]["n_papers"],
                "relevant": is_relevant
            })

        # Calculate baseline metrics for this query
        p5, p10, rr, rel10 = calculate_metrics_for_ranking(query_labels)
        baseline_metrics.append({
            "query": query,
            "Precision@5": round(p5, 4),
            "Precision@10": round(p10, 4),
            "Reciprocal_Rank": round(rr, 4),
            "Relevant@10": round(rel10, 1)
        })

    # Save candidates
    df_candidates = pd.DataFrame(candidate_rows)
    df_candidates.to_csv(OUTPUT_CANDIDATES, index=False)
    print(f"Saved {len(df_candidates)} candidates to: {OUTPUT_CANDIDATES}")

    # Save labeled benchmark
    df_labeled = pd.DataFrame(labeled_rows)
    df_labeled.to_csv(OUTPUT_LABELED, index=False)
    print(f"Saved {len(df_labeled)} labeled annotations to: {OUTPUT_LABELED}")

    # Save baseline metrics
    df_metrics = pd.DataFrame(baseline_metrics)
    mean_row = {
        "query": "MEAN",
        "Precision@5": round(df_metrics["Precision@5"].mean(), 4),
        "Precision@10": round(df_metrics["Precision@10"].mean(), 4),
        "Reciprocal_Rank": round(df_metrics["Reciprocal_Rank"].mean(), 4),
        "Relevant@10": round(df_metrics["Relevant@10"].mean(), 1)
    }
    df_metrics = pd.concat([df_metrics, pd.DataFrame([mean_row])], ignore_index=True)
    df_metrics.to_csv(OUTPUT_BASELINE_METRICS, index=False)
    print(f"Saved baseline metrics to: {OUTPUT_BASELINE_METRICS}")

    print("\n" + "=" * 70)
    print("BASELINE MODEL EVALUATION SUMMARY")
    print("=" * 70)
    print(df_metrics.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    generate_and_annotate_benchmark()