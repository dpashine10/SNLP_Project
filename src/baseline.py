import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = PROJECT_ROOT / "Data/processed/researcher_profiles.csv"


def main():
    # Load researcher profiles
    df = pd.read_csv(PROFILE_PATH, keep_default_na=False)

    print(f"Loaded {len(df)} researcher profiles.")

    # TF-IDF baseline
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True
    )

    profile_vectors = vectorizer.fit_transform(
        df["profile_text"]
    )

    print(f"TF-IDF matrix shape: {profile_vectors.shape}")

    print("\nEnter a research topic/query.")
    print("Type 'exit' to stop.\n")

    while True:
        query = input("Query: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a query.\n")
            continue

        # Convert query to TF-IDF
        query_vector = vectorizer.transform([query])

        # Cosine similarity
        scores = cosine_similarity(
            query_vector,
            profile_vectors
        ).flatten()

        # Top 10 researchers
        top_indices = scores.argsort()[::-1][:10]

        print("\nTop researchers:")
        print("-" * 80)

        for rank, idx in enumerate(top_indices, start=1):
            print(
                f"{rank}. {df.iloc[idx]['author_name']}"
                f" | Score: {scores[idx]:.4f}"
                f" | Papers: {df.iloc[idx]['n_papers']}"
            )

        print()


if __name__ == "__main__":
    main()