import pandas as pd
from pathlib import Path
import os

os.chdir("/Users/divy/College/NLP/Project")
PAPERS_PATH = "data/processed/papers.csv"
AUTHORS_PATH = "data/processed/paper_author.csv"
OUTPUT_PATH = "data/processed/researcher_profiles.csv"


def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    papers = pd.read_csv(
        PAPERS_PATH,
        keep_default_na=False
    )

    authors = pd.read_csv(
        AUTHORS_PATH,
        keep_default_na=False
    )

    # Keep only valid researcher IDs
    authors = authors[
        (authors["author_id"] != "") &
        (authors["author_name"] != "")
    ].copy()

    # Merge researcher-paper relationships with paper information
    merged = authors.merge(
        papers,
        on="work_id",
        how="left"
    )

    # Make sure text fields are strings
    text_columns = [
        "title",
        "abstract",
        "topics",
        "keywords"
    ]

    for col in text_columns:
        merged[col] = merged[col].fillna("").astype(str)

    # Build text representation for each paper
    merged["paper_text"] = (
        merged["title"] + " "
        + merged["abstract"] + " "
        + merged["topics"] + " "
        + merged["keywords"]
    ).str.strip()

    # Remove completely empty paper text
    merged = merged[
        merged["paper_text"] != ""
    ]

    # Group all research work belonging to each researcher
    profiles = (
        merged
        .groupby(["author_id", "author_name"])
        .agg(
            n_papers=("work_id", "nunique"),
            paper_ids=("work_id", lambda x: "|".join(x.astype(str).unique())),
            profile_text=("paper_text", lambda x: " ".join(x))
        )
        .reset_index()
    )

    # Remove duplicate whitespace
    profiles["profile_text"] = (
        profiles["profile_text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    profiles.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("Researcher profiles created")
    print("--------------------------------")
    print("Profiles:", len(profiles))
    print("Researchers with 1 paper:",
          (profiles["n_papers"] == 1).sum())
    print("Researchers with >1 paper:",
          (profiles["n_papers"] > 1).sum())

    print("\nAverage papers per researcher:",
          profiles["n_papers"].mean())

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("\nSample profiles:")
    print(
        profiles[
            ["author_id", "author_name", "n_papers", "profile_text"]
        ].head(5).to_string()
    )


if __name__ == "__main__":
    main()