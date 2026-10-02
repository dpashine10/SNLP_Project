"""
Build RAG-Ready Datasets and Researcher Profiles.
Applies transformations from src/buildprofiles.py and src/build_embeddings.py
to the new processed dataset (Data/indian_institutions_works_processed.csv).

Generates:
  1. Data/processed/researcher_profiles.csv (Researcher profiles with clean profile_text)
  2. Data/processed/papers.csv (Deduplicated unique papers with synthesized paper_text)
  3. Data/processed/paper_author.csv (Relational author-paper mapping)
  4. Data/processed/rag_documents.csv (Simplified RAG corpus ready for Vector DB ingestion)
"""

import json
from pathlib import Path
import time
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "Data" / "indian_institutions_works_processed.csv"
PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"

PAPERS_PATH = PROCESSED_DIR / "papers.csv"
AUTHORS_PATH = PROCESSED_DIR / "paper_author.csv"
PROFILES_PATH = PROCESSED_DIR / "researcher_profiles.csv"
RAG_DOCS_PATH = PROCESSED_DIR / "rag_documents.csv"


def join_unique_strings(series: pd.Series, sep: str = "; ") -> str:
    """Merges unique non-empty strings."""
    vals = [
        str(x).strip()
        for x in series.dropna().unique()
        if str(x).strip() and str(x).strip().lower() != "nan"
    ]
    return sep.join(vals) if vals else ""


def build_rag_datasets(input_csv: Path = INPUT_PATH):
    start_time = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading preprocessed dataset from: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    total_records = len(df)
    print(f"Loaded {total_records:,} records.")

    # 1. Clean and normalize text columns
    print("Normalizing text columns...")
    for col in ["title", "abstract", "topics", "keywords", "institution_name", "author_name", "doi"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Clean topics & keywords formatting
    df["clean_topics"] = (
        df["topics"]
        .str.replace("|", ", ", regex=False)
        .str.replace(";", ", ", regex=False)
        .str.strip()
    )
    df["clean_keywords"] = (
        df["keywords"]
        .str.replace("|", ", ", regex=False)
        .str.replace(";", ", ", regex=False)
        .str.strip()
    )

    # 2. Build rich paper_text representation (as in build_embeddings.py & buildprofiles.py)
    print("Formulating rich semantic paper_text...")
    paper_text_parts = [df["title"]]
    
    # Title + Abstract
    has_abstract = df["abstract"] != ""
    df["paper_text"] = df["title"]
    df.loc[has_abstract, "paper_text"] = df.loc[has_abstract, "title"] + ". " + df.loc[has_abstract, "abstract"]
    
    # + Topics
    has_topics = df["clean_topics"] != ""
    df.loc[has_topics, "paper_text"] = df.loc[has_topics, "paper_text"] + " Topics: " + df.loc[has_topics, "clean_topics"]
    
    # + Keywords
    has_keywords = df["clean_keywords"] != ""
    df.loc[has_keywords, "paper_text"] = df.loc[has_keywords, "paper_text"] + " Keywords: " + df.loc[has_keywords, "clean_keywords"]
    
    # Collapse multiple whitespaces
    df["paper_text"] = df["paper_text"].str.replace(r"\s+", " ", regex=True).str.strip()

    # Fallback to title if empty
    empty_mask = df["paper_text"] == ""
    df.loc[empty_mask, "paper_text"] = df.loc[empty_mask, "title"]

    # ==========================================
    # 3. Generate Data/processed/papers.csv (Unique papers)
    # ==========================================
    print("Generating unique papers dataset (Data/processed/papers.csv)...")
    papers_df = df.groupby("work_id", as_index=False).agg({
        "title": "first",
        "abstract": "first",
        "clean_topics": "first",
        "clean_keywords": "first",
        "publication_year": "first",
        "doi": "first",
        "paper_text": "first",
        "author_name": lambda s: join_unique_strings(s, sep="; "),
        "author_id": lambda s: join_unique_strings(s, sep="|"),
        "institution_name": lambda s: join_unique_strings(s, sep="; ")
    }).rename(columns={
        "clean_topics": "topics",
        "clean_keywords": "keywords",
        "author_name": "authors",
        "author_id": "author_ids",
        "institution_name": "institutions"
    })

    papers_df.to_csv(PAPERS_PATH, index=False)
    print(f"  -> Saved {len(papers_df):,} unique papers to: {PAPERS_PATH}")

    # ==========================================
    # 4. Generate Data/processed/paper_author.csv (Relational Mapping)
    # ==========================================
    print("Generating author-paper relational mapping (Data/processed/paper_author.csv)...")
    paper_author_df = df[
        df["author_id"].notna() & (df["author_id"].str.strip() != "")
    ][["work_id", "author_id", "author_name", "institution_name"]].drop_duplicates()
    paper_author_df.to_csv(AUTHORS_PATH, index=False)
    print(f"  -> Saved {len(paper_author_df):,} author-paper links to: {AUTHORS_PATH}")

    # ==========================================
    # 5. Generate Data/processed/researcher_profiles.csv (Researcher Profiles)
    # ==========================================
    print("Generating researcher profiles (Data/processed/researcher_profiles.csv)...")
    valid_authors_df = df[
        df["author_id"].notna() & 
        (df["author_id"].str.strip() != "") & 
        (df["author_name"].str.strip() != "") &
        (df["paper_text"] != "")
    ]

    profiles_df = valid_authors_df.groupby(["author_id", "author_name"], as_index=False).agg(
        institution_name=("institution_name", lambda s: join_unique_strings(s, sep="; ")),
        n_papers=("work_id", "nunique"),
        paper_ids=("work_id", lambda x: "|".join(x.astype(str).unique())),
        profile_text=("paper_text", lambda x: " ".join(x))
    )

    # Clean profile_text whitespace
    profiles_df["profile_text"] = (
        profiles_df["profile_text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    profiles_df.to_csv(PROFILES_PATH, index=False)
    print(f"  -> Saved {len(profiles_df):,} researcher profiles to: {PROFILES_PATH}")

    # ==========================================
    # 6. Generate Data/processed/rag_documents.csv (Simple RAG Corpus)
    # ==========================================
    print("Generating simplified RAG corpus (Data/processed/rag_documents.csv)...")
    # For RAG, each document is a self-contained record ready for any Vector DB
    rag_docs = []
    
    # 6a. Add unique papers as RAG documents
    for _, row in papers_df.iterrows():
        meta = {
            "entity_type": "paper",
            "title": row["title"],
            "authors": row["authors"],
            "institutions": row["institutions"],
            "year": str(row["publication_year"]),
            "doi": str(row["doi"]),
            "topics": row["topics"]
        }
        rag_docs.append({
            "doc_id": row["work_id"],
            "entity_type": "paper",
            "title_or_name": row["title"],
            "text": row["paper_text"],
            "metadata_json": json.dumps(meta)
        })

    # 6b. Add researcher profiles as RAG documents
    for _, row in profiles_df.iterrows():
        meta = {
            "entity_type": "researcher",
            "author_name": row["author_name"],
            "institutions": row["institution_name"],
            "n_papers": int(row["n_papers"])
        }
        rag_docs.append({
            "doc_id": row["author_id"],
            "entity_type": "researcher",
            "title_or_name": row["author_name"],
            "text": row["profile_text"][:2000],  # truncated summary representation for vector indexing
            "metadata_json": json.dumps(meta)
        })

    rag_df = pd.DataFrame(rag_docs)
    rag_df.to_csv(RAG_DOCS_PATH, index=False)
    print(f"  -> Saved {len(rag_df):,} simplified RAG documents to: {RAG_DOCS_PATH}")

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Transformation Pipeline Completed in {elapsed:.2f} seconds!")
    print(f"Generated Outputs:")
    print(f"  1. Papers:              {PAPERS_PATH} ({len(papers_df):,} rows)")
    print(f"  2. Paper-Author Links:  {AUTHORS_PATH} ({len(paper_author_df):,} rows)")
    print(f"  3. Researcher Profiles: {PROFILES_PATH} ({len(profiles_df):,} rows)")
    print(f"  4. Simplified RAG Docs: {RAG_DOCS_PATH} ({len(rag_df):,} rows)")
    print("-" * 60)


if __name__ == "__main__":
    build_rag_datasets()