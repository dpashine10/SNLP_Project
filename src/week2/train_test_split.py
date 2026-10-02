"""
Train/Test Split Script.
Splits papers and researcher profiles into 80% Train and 20% Held-out Test sets
for model benchmarking, vector store evaluation, and RAG validation.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPERS_PATH = PROJECT_ROOT / "Data" / "processed" / "papers.csv"
PROFILES_PATH = PROJECT_ROOT / "Data" / "processed" / "researcher_profiles.csv"

TRAIN_PAPERS_PATH = PROJECT_ROOT / "Data" / "processed" / "train_papers.csv"
TEST_PAPERS_PATH = PROJECT_ROOT / "Data" / "processed" / "test_papers.csv"

TRAIN_PROFILES_PATH = PROJECT_ROOT / "Data" / "processed" / "train_profiles.csv"
TEST_PROFILES_PATH = PROJECT_ROOT / "Data" / "processed" / "test_profiles.csv"


def split_datasets(test_size: float = 0.2, random_state: int = 42):
    print("=" * 60)
    print("EXECUTING TRAIN/TEST SPLIT (80% TRAIN / 20% TEST)")
    print("=" * 60)

    # 1. Split Papers
    if PAPERS_PATH.exists():
        print(f"Loading papers from: {PAPERS_PATH}")
        papers_df = pd.read_csv(PAPERS_PATH, low_memory=False)
        total_papers = len(papers_df)

        train_papers, test_papers = train_test_split(
            papers_df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        train_papers.to_csv(TRAIN_PAPERS_PATH, index=False)
        test_papers.to_csv(TEST_PAPERS_PATH, index=False)

        print(f"Papers Split:")
        print(f"  - Total: {total_papers:,}")
        print(f"  - Train (80%): {len(train_papers):,} -> {TRAIN_PAPERS_PATH.name}")
        print(f"  - Test  (20%): {len(test_papers):,} -> {TEST_PAPERS_PATH.name}")
    else:
        print(f"Warning: {PAPERS_PATH} not found.")

    # 2. Split Profiles
    if PROFILES_PATH.exists():
        print(f"\nLoading researcher profiles from: {PROFILES_PATH}")
        profiles_df = pd.read_csv(PROFILES_PATH, low_memory=False)
        total_profiles = len(profiles_df)

        train_profiles, test_profiles = train_test_split(
            profiles_df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        train_profiles.to_csv(TRAIN_PROFILES_PATH, index=False)
        test_profiles.to_csv(TEST_PROFILES_PATH, index=False)

        print(f"Researcher Profiles Split:")
        print(f"  - Total: {total_profiles:,}")
        print(f"  - Train (80%): {len(train_profiles):,} -> {TRAIN_PROFILES_PATH.name}")
        print(f"  - Test  (20%): {len(test_profiles):,} -> {TEST_PROFILES_PATH.name}")
    else:
        print(f"Warning: {PROFILES_PATH} not found.")

    print("\nTrain/Test Split successfully completed!")


if __name__ == "__main__":
    split_datasets()
