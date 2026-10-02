"""
Data Preprocessing Script: Author-Paper Deduplication.
Deduplicates works at the author-paper level and merges cross-institution affiliations.
Input:  Data/indian_institutions_works.csv
Output: Data/indian_institutions_works_processed.csv
"""

from pathlib import Path
import time
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "Data" / "indian_institutions_works.csv"
OUTPUT_CSV = PROJECT_ROOT / "Data" / "indian_institutions_works_processed.csv"


def get_author_key(row) -> str:
    """Generates a stable key for an author on a paper."""
    aid = str(row["author_id"]).strip() if pd.notna(row["author_id"]) else ""
    if aid and aid.lower() != "nan":
        return f"id:{aid}"
    aname = str(row["author_name"]).strip().lower() if pd.notna(row["author_name"]) else ""
    if aname and aname != "nan":
        return f"name:{aname}"
    return "__unknown_author__"


def join_unique_strings(series: pd.Series) -> str:
    """Merges unique non-empty strings separated by semicolons."""
    vals = [
        str(x).strip()
        for x in series.dropna().unique()
        if str(x).strip() and str(x).strip().lower() != "nan"
    ]
    return "; ".join(vals) if vals else ""


def deduplicate_author_works(input_path: Path = INPUT_CSV, output_path: Path = OUTPUT_CSV):
    start_time = time.time()
    print(f"Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    initial_rows = len(df)
    print(f"Total rows loaded: {initial_rows:,}")

    # Build unique author-paper composite key
    print("Generating author-paper composite deduplication keys...")
    df["dedup_key"] = df["work_id"].astype(str) + "___" + df.apply(get_author_key, axis=1)

    # Separate completely unique rows from duplicates for maximum speed
    is_dup = df.duplicated("dedup_key", keep=False)
    unique_df = df[~is_dup].copy()
    dup_df = df[is_dup].copy()

    dup_groups_count = dup_df["dedup_key"].nunique()
    print(f"Identified {len(unique_df):,} single-affiliation author-paper rows.")
    print(f"Identified {len(dup_df):,} multi-institution/duplicate rows across {dup_groups_count:,} unique author-paper pairs.")

    # Merge duplicate groups while aggregating cross-institutional affiliations
    if not dup_df.empty:
        merged_dups = dup_df.groupby("dedup_key", as_index=False).agg({
            "institution_id": join_unique_strings,
            "institution_name": join_unique_strings,
            "work_id": "first",
            "author_id": "first",
            "author_name": "first",
            "title": "first",
            "abstract": "first",
            "topics": "first",
            "keywords": "first",
            "publication_year": "first",
            "doi": "first"
        })
    else:
        merged_dups = pd.DataFrame()

    # Combine unique and merged records
    final_df = pd.concat(
        [unique_df.drop(columns=["dedup_key"]), merged_dups.drop(columns=["dedup_key"])],
        ignore_index=True
    )

    # Ensure identical column ordering
    col_order = [
        "institution_id", "institution_name", "work_id", "author_id", "author_name",
        "title", "abstract", "topics", "keywords", "publication_year", "doi"
    ]
    final_df = final_df[col_order]

    final_rows = len(final_df)
    duplicates_removed = initial_rows - final_rows

    print(f"Writing deduplicated dataset to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - start_time

    print(f"Successfully processed in {elapsed:.2f} seconds!")
    print(f"Summary:")
    print(f"  - Initial rows:       {initial_rows:,}")
    print(f"  - Duplicates merged:  {duplicates_removed:,}")
    print(f"  - Final rows:         {final_rows:,}")
    print(f"  - Output size:        {file_size_mb:.2f} MB")

    return final_df


if __name__ == "__main__":
    deduplicate_author_works()
