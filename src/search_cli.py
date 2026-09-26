"""
Interactive CLI for the Research Discovery & Collaboration Engine (Version 1 NLP Pipeline).
Allows querying researchers using Dense Semantic Similarity, TF-IDF Baseline, or Hybrid search.
Outputs researcher names, similarity scores, most relevant paper, and related topics.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import ResearchDiscoveryPipeline


def print_header():
    print("=" * 80)
    print("🎓 AI UNIVERSITY - RESEARCH DISCOVERY & COLLABORATION ENGINE")
    print("   Version 1.0 NLP Pipeline (Dense Semantic Embeddings + TF-IDF Baseline)")
    print("=" * 80)


def print_result(res: dict):
    rank = res["rank"]
    name = res["author_name"]
    score = res["similarity_score"]
    n_papers = res["n_papers"]
    paper = res["relevant_paper"]
    topics = res["related_topics"]

    print(f"\n[{rank}] Researcher: {name}")
    print(f"    Similarity Score: {score:.4f}  |  Total Publications: {n_papers}")
    print(f"    Relevant Paper:   \"{paper['title']}\" ({paper.get('year', 'N/A')})")
    if paper.get("paper_score"):
        print(f"    Paper Similarity: {paper['paper_score']:.4f}")
    if topics:
        print(f"    Related Topics:   {', '.join(topics[:4])}")


def main():
    print_header()
    print("Initializing pipeline...")
    pipeline = ResearchDiscoveryPipeline()

    print("\nSelect search mode:")
    print("  [1] Semantic (Dense Sentence-BERT embeddings - Recommended)")
    print("  [2] TF-IDF (Sparse Keyword Baseline)")
    print("  [3] Hybrid (Dense Semantic + TF-IDF Keyword Blending)")

    mode_choice = input("\nChoose mode [1/2/3] (default: 1): ").strip()
    mode_map = {"1": "semantic", "2": "tfidf", "3": "hybrid", "": "semantic"}
    mode = mode_map.get(mode_choice, "semantic")

    print(f"\nActive Mode: {mode.upper()}")
    print("-" * 80)
    print("Enter a research topic, problem description, or paper abstract.")
    print("Type 'mode' to switch retrieval mode, or 'exit' to quit.\n")

    while True:
        try:
            query = input("\nResearch Query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if query.lower() == "mode":
            print("\nSelect search mode:")
            print("  [1] Semantic")
            print("  [2] TF-IDF")
            print("  [3] Hybrid")
            m = input("Choice: ").strip()
            mode = mode_map.get(m, mode)
            print(f"Switched mode to: {mode.upper()}")
            continue

        if not query:
            print("Please enter a research topic or description.")
            continue

        print(f"\nSearching with [{mode.upper()}]...")
        results = pipeline.search(query=query, top_k=10, mode=mode)

        if not results:
            print("No matching researchers found.")
            continue

        print(f"\nTop {len(results)} Researcher Matches:")
        print("-" * 80)
        for res in results:
            print_result(res)
        print("-" * 80)


if __name__ == "__main__":
    main()
