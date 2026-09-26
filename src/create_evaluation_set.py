import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
os.chdir("/Users/divy/College/NLP/Project")
PROFILE_PATH = "Data/processed/researcher_profiles.csv"
OUTPUT_PATH = "Data/processed/evaluation_candidates.csv"

QUERIES = [
    "natural language processing",
    "machine learning",
    "transformer based text classification",
    "information retrieval",
    "knowledge graphs",
]

TOP_K = 10


def main():
    df = pd.read_csv(
        PROFILE_PATH,
        keep_default_na=False
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000
    )

    profile_vectors = vectorizer.fit_transform(
        df["profile_text"]
    )

    rows = []

    for query in QUERIES:
        query_vector = vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            profile_vectors
        ).flatten()

        top_indices = scores.argsort()[::-1][:TOP_K]

        for rank, idx in enumerate(top_indices, start=1):
            rows.append({
                "query": query,
                "rank": rank,
                "author_id": df.iloc[idx]["author_id"],
                "author_name": df.iloc[idx]["author_name"],
                "score": round(float(scores[idx]), 4),
                "n_papers": df.iloc[idx]["n_papers"],
                "relevant": ""
            })

    results = pd.DataFrame(rows)

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Created: {OUTPUT_PATH}")
    print(f"Rows: {len(results)}")


if __name__ == "__main__":
    main()