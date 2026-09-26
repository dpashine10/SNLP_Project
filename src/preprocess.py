from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "Data/raw/rawdata.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data/processed"


def main() -> None:
    raw = pd.read_csv(RAW_PATH, keep_default_na=False)
    raw = raw.fillna("")

    required_columns = {
        "work_id",
        "author_id",
        "author_name",
        "title",
        "abstract",
        "topics",
        "keywords",
        "publication_year",
    }
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    raw = raw[raw["work_id"].astype(str).str.strip() != ""].copy()
    text_columns = ["title", "abstract", "topics", "keywords"]
    for column in text_columns:
        raw[column] = raw[column].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    papers = raw[
        ["work_id", "title", "abstract", "topics", "keywords", "publication_year"]
    ].drop_duplicates("work_id", keep="first")

    paper_authors = raw[["work_id", "author_id", "author_name"]].copy()
    paper_authors["author_id"] = paper_authors["author_id"].astype(str).str.strip()
    paper_authors["author_name"] = paper_authors["author_name"].astype(str).str.strip()
    paper_authors = paper_authors[
        paper_authors["author_id"].ne("") & paper_authors["author_name"].ne("")
    ].drop_duplicates(["work_id", "author_id"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    papers.to_csv(OUTPUT_DIR / "papers.csv", index=False)
    paper_authors.to_csv(OUTPUT_DIR / "paper_author.csv", index=False)

    print(f"Raw rows: {len(raw)}")
    print(f"Unique papers: {papers['work_id'].nunique()}")
    print(f"Valid paper-author relationships: {len(paper_authors)}")
    print(f"Unique researchers: {paper_authors['author_id'].nunique()}")
    print(f"Papers missing abstracts: {papers['abstract'].eq('').sum()}")
    print(f"Papers missing publication year: {papers['publication_year'].eq('').sum()}")
    print(f"Raw rows missing author IDs: {raw['author_id'].eq('').sum()}")
    print(f"Duplicate raw relationships removed: {raw.duplicated(['work_id', 'author_id']).sum()}")


if __name__ == "__main__":
    main()