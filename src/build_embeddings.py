"""
Precompute and cache dense sentence embeddings for papers and researcher profiles
using Sentence Transformers (all-MiniLM-L6-v2).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_PATH = PROJECT_ROOT / "Data" / "processed" / "papers.csv"
AUTHORS_PATH = PROJECT_ROOT / "Data" / "processed" / "paper_author.csv"
PROFILES_PATH = PROJECT_ROOT / "Data" / "processed" / "researcher_profiles.csv"

PAPER_EMB_PATH = PROJECT_ROOT / "Data" / "processed" / "paper_embeddings.npz"
RESEARCHER_EMB_PATH = PROJECT_ROOT / "Data" / "processed" / "researcher_embeddings.npz"

MODEL_NAME = "all-MiniLM-L6-v2"


def build_embeddings():
    print(f"Loading SentenceTransformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Loading papers from: {PAPERS_PATH}")
    papers_df = pd.read_csv(PAPERS_PATH, keep_default_na=False)

    # Clean text representation for each paper
    # Using title, abstract, topics, keywords
    papers_df["clean_title"] = papers_df["title"].fillna("").astype(str).str.strip()
    papers_df["clean_abstract"] = papers_df["abstract"].fillna("").astype(str).str.strip()
    papers_df["clean_topics"] = papers_df["topics"].fillna("").astype(str).str.replace("|", ", ", regex=False)
    papers_df["clean_keywords"] = papers_df["keywords"].fillna("").astype(str).str.replace("|", ", ", regex=False)

    papers_df["paper_text"] = (
        papers_df["clean_title"] + ". "
        + papers_df["clean_abstract"] + " Topics: "
        + papers_df["clean_topics"] + " Keywords: "
        + papers_df["clean_keywords"]
    ).str.strip()

    # If title & abstract empty, fallback to title
    papers_df.loc[papers_df["paper_text"] == "", "paper_text"] = papers_df["clean_title"]

    work_ids = papers_df["work_id"].values
    texts = papers_df["paper_text"].tolist()

    print(f"Encoding {len(texts)} papers in batches...")
    paper_embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype(np.float32)

    np.savez_compressed(
        PAPER_EMB_PATH,
        work_ids=work_ids,
        embeddings=paper_embeddings
    )
    print(f"Saved paper embeddings to: {PAPER_EMB_PATH}")

    # Map work_id -> paper embedding
    work_to_emb = {wid: paper_embeddings[i] for i, wid in enumerate(work_ids)}

    # Build researcher embeddings
    print(f"Loading researcher profiles from: {PROFILES_PATH}")
    profiles_df = pd.read_csv(PROFILES_PATH, keep_default_na=False)

    researcher_author_ids = []
    researcher_embeddings = []

    fallback_texts = []
    fallback_indices = []

    for idx, row in tqdm(profiles_df.iterrows(), total=len(profiles_df), desc="Aggregating researcher profiles"):
        author_id = row["author_id"]
        paper_ids = str(row["paper_ids"]).split("|") if row["paper_ids"] else []

        embs = [work_to_emb[pid] for pid in paper_ids if pid in work_to_emb]

        if embs:
            # Mean pooling across authored papers
            mean_emb = np.mean(embs, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm
            researcher_embeddings.append(mean_emb)
            researcher_author_ids.append(author_id)
        else:
            # Fallback to profile_text
            fallback_texts.append(str(row["profile_text"])[:512])
            fallback_indices.append((len(researcher_author_ids), author_id))
            researcher_author_ids.append(author_id)
            researcher_embeddings.append(None)

    if fallback_texts:
        print(f"Encoding {len(fallback_texts)} fallback profiles...")
        fb_embs = model.encode(
            fallback_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True
        ).astype(np.float32)
        for (pos, _), fb_emb in zip(fallback_indices, fb_embs):
            researcher_embeddings[pos] = fb_emb

    researcher_embeddings = np.array(researcher_embeddings, dtype=np.float32)

    np.savez_compressed(
        RESEARCHER_EMB_PATH,
        author_ids=np.array(researcher_author_ids),
        embeddings=researcher_embeddings
    )
    print(f"Saved researcher embeddings to: {RESEARCHER_EMB_PATH}")
    print(f"Shape: {researcher_embeddings.shape}")
    print("Precomputation complete!")


if __name__ == "__main__":
    build_embeddings()
