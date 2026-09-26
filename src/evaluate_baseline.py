from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPERS_PATH = PROJECT_ROOT / "Data/processed/papers.csv"
AUTHORS_PATH = PROJECT_ROOT / "Data/processed/paper_author.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data/processed/calculated"
RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_K = 10


def build_profiles(papers: pd.DataFrame, authors: pd.DataFrame) -> pd.DataFrame:
    merged = authors.merge(papers, on="work_id", how="inner")
    text_columns = ["title", "abstract", "topics", "keywords"]
    for column in text_columns:
        merged[column] = merged[column].fillna("").astype(str)
    merged["paper_text"] = merged[text_columns].agg(" ".join, axis=1).str.replace(
        r"\s+", " ", regex=True
    ).str.strip()
    merged = merged[merged["paper_text"] != ""]
    return (
        merged.groupby("author_id", as_index=False)
        .agg(
            author_name=("author_name", "first"),
            n_papers=("work_id", "nunique"),
            profile_text=("paper_text", " ".join),
        )
    )


def main() -> None:
    papers = pd.read_csv(PAPERS_PATH, keep_default_na=False)
    authors = pd.read_csv(AUTHORS_PATH, keep_default_na=False)
    authors = authors[(authors["author_id"] != "") & (authors["author_name"] != "")]

    work_ids = papers["work_id"].drop_duplicates().sort_values()
    train_ids, test_ids = train_test_split(
        work_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_ids = set(train_ids)
    test_ids = set(test_ids)

    train_papers = papers[papers["work_id"].isin(train_ids)]
    test_papers = papers[papers["work_id"].isin(test_ids)]
    train_authors = authors[authors["work_id"].isin(train_ids)]
    test_authors = authors[authors["work_id"].isin(test_ids)]
    train_profiles = build_profiles(train_papers, train_authors)
    known_authors = set(train_profiles["author_id"])
    paper_counts = authors.groupby("author_id")["work_id"].nunique()
    established_authors = set(paper_counts[paper_counts >= 2].index)

    test_queries = test_papers.merge(test_authors, on="work_id", how="inner")
    test_queries = test_queries[test_queries["author_id"].isin(known_authors)].copy()
    text_columns = ["title", "abstract", "topics", "keywords"]
    for column in text_columns:
        test_queries[column] = test_queries[column].fillna("").astype(str)
    test_queries["query"] = test_queries[text_columns].agg(" ".join, axis=1).str.replace(
        r"\s+", " ", regex=True
    ).str.strip()
    relevant = test_queries.groupby("work_id")["author_id"].agg(set).to_dict()
    query_rows = test_queries[["work_id", "query"]].drop_duplicates()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
    )
    profile_vectors = vectorizer.fit_transform(train_profiles["profile_text"])
    query_vectors = vectorizer.transform(query_rows["query"])
    scores = cosine_similarity(query_vectors, profile_vectors)

    candidate_rows = []
    metric_rows = []
    for row_index, (_, query_row) in enumerate(query_rows.iterrows()):
        top_indices = scores[row_index].argsort()[::-1][:TOP_K]
        labels = [
            int(train_profiles.iloc[index]["author_id"] in relevant[query_row["work_id"]])
            for index in top_indices
        ]
        for rank, (index, label) in enumerate(zip(top_indices, labels), start=1):
            profile = train_profiles.iloc[index]
            candidate_rows.append(
                {
                    "work_id": query_row["work_id"],
                    "query": query_row["query"],
                    "rank": rank,
                    "author_id": profile["author_id"],
                    "author_name": profile["author_name"],
                    "score": round(float(scores[row_index, index]), 6),
                    "relevant": label,
                }
            )
        first_relevant = next(
            (rank for rank, label in enumerate(labels, start=1) if label), 0
        )
        metric_rows.append(
            {
                "work_id": query_row["work_id"],
                "Precision@5": sum(labels[:5]) / 5,
                "Precision@10": sum(labels) / TOP_K,
                "Recall@10": sum(labels) / len(relevant[query_row["work_id"]]),
                "Reciprocal_Rank": 1 / first_relevant if first_relevant else 0.0,
                "Hit@10": int(any(labels)),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"work_id": sorted(train_ids), "split": "train"}).to_csv(
        OUTPUT_DIR / "train_papers.csv", index=False
    )
    pd.DataFrame({"work_id": sorted(test_ids), "split": "test"}).to_csv(
        OUTPUT_DIR / "test_papers.csv", index=False
    )
    query_rows.to_csv(OUTPUT_DIR / "heldout_queries.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(
        OUTPUT_DIR / "heldout_evaluation_candidates.csv", index=False
    )
    metrics = pd.DataFrame(metric_rows)
    metrics.loc[len(metrics)] = {
        "work_id": "MEAN",
        "Precision@5": metrics["Precision@5"].mean(),
        "Precision@10": metrics["Precision@10"].mean(),
        "Recall@10": metrics["Recall@10"].mean(),
        "Reciprocal_Rank": metrics["Reciprocal_Rank"].mean(),
        "Hit@10": metrics["Hit@10"].mean(),
    }
    metrics.to_csv(OUTPUT_DIR / "heldout_metrics.csv", index=False)

    # Report the non-cold-start slice separately. This is not a replacement for
    # the overall result; it measures the intended profile-matching use case.
    established_profiles = train_profiles[
        train_profiles["author_id"].isin(established_authors)
    ].reset_index(drop=True)
    established_vectors = vectorizer.transform(established_profiles["profile_text"])
    established_scores = cosine_similarity(query_vectors, established_vectors)
    established_metric_rows = []
    for row_index, (_, query_row) in enumerate(query_rows.iterrows()):
        top_indices = established_scores[row_index].argsort()[::-1][:TOP_K]
        labels = [
            int(
                established_profiles.iloc[index]["author_id"]
                in relevant[query_row["work_id"]]
            )
            for index in top_indices
        ]
        first_relevant = next(
            (rank for rank, label in enumerate(labels, start=1) if label), 0
        )
        established_metric_rows.append(
            {
                "work_id": query_row["work_id"],
                "Precision@5": sum(labels[:5]) / 5,
                "Precision@10": sum(labels) / TOP_K,
                "Recall@10": sum(labels) / len(relevant[query_row["work_id"]]),
                "Reciprocal_Rank": 1 / first_relevant if first_relevant else 0.0,
                "Hit@10": int(any(labels)),
            }
        )
    established_metrics = pd.DataFrame(established_metric_rows)
    established_metrics.loc[len(established_metrics)] = {
        "work_id": "MEAN",
        "Precision@5": established_metrics["Precision@5"].mean(),
        "Precision@10": established_metrics["Precision@10"].mean(),
        "Recall@10": established_metrics["Recall@10"].mean(),
        "Reciprocal_Rank": established_metrics["Reciprocal_Rank"].mean(),
        "Hit@10": established_metrics["Hit@10"].mean(),
    }
    established_metrics.to_csv(
        OUTPUT_DIR / "established_researcher_metrics.csv", index=False
    )

    print(f"Train papers: {len(train_ids)}")
    print(f"Test papers: {len(test_ids)}")
    print(f"Held-out queries with known authors: {len(query_rows)}")
    print(f"Train researcher profiles: {len(train_profiles)}")
    print("Mean metrics:")
    print(metrics.tail(1).to_string(index=False))
    print("Established-researcher mean metrics:")
    print(established_metrics.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()