"""Command-line interface (CLI) for the Reusable NLP Asset.

Usage::

    # Search researchers
    python -m src.week6.cli --query "graph neural networks" --top-k 5

    # Select ranking strategy
    python -m src.week6.cli -q "medical imaging transformers" -s best_paper

    # Output machine-readable JSON
    python -m src.week6.cli -q "reinforcement learning" --json

    # Inspect asset health and metadata
    python -m src.week6.cli --health
    python -m src.week6.cli --info
"""

from __future__ import annotations

import argparse
import json
import sys

from src.week6.asset import AssetConfig, NLPAsset
from src.week6.schemas import (
    DEFAULT_STRATEGY,
    DEFAULT_TOP_K,
    SUPPORTED_STRATEGIES,
    RecommendRequest,
)


def _print_table(response_dict: dict) -> None:
    results = response_dict.get("results", [])
    query = response_dict.get("query", "")
    time_ms = response_dict.get("execution_time_ms", 0.0)
    backend = response_dict.get("model_info", {}).get("backend", "")
    strategy = response_dict.get("model_info", {}).get("strategy", "")

    print("\n" + "=" * 80)
    print(f" RESEARCHER RECOMMENDATIONS (Query: '{query}')")
    print(f" Backend: {backend} | Strategy: {strategy} | Latency: {time_ms:.1f}ms")
    print("=" * 80)

    if not results:
        print(" No matching researchers found above confidence threshold.")
        print("=" * 80)
        return

    header = f"{'Rank':<5} {'Name':<24} {'Score':<8} {'Conf':<8} {'Confidence Level & Reason':<32}"
    print(header)
    print("-" * 80)

    for r in results:
        rank = r["rank"]
        name = r["name"][:23]
        score = f"{r['score']:.4f}"
        conf_score = f"{r['confidence']['score']:.2f}"
        conf_level = r["confidence"]["level"]
        line = f"[{rank}]   {name:<24} {score:<8} {conf_score:<8} [{conf_level}]"
        print(line)

        evidence = r.get("evidence", [])
        if evidence:
            top_paper = evidence[0]
            title = top_paper["title"][:60]
            year = top_paper["year"]
            sim = top_paper["similarity"]
            print(f"      |-- Top Paper ({year}) [sim={sim:.2f}]: \"{title}\"")
        print()

    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI tool for the Reusable NLP Researcher Recommendation Asset."
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Research topic, problem synopsis, or natural language query.",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of researchers to return (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "-s", "--strategy",
        type=str,
        default=DEFAULT_STRATEGY,
        choices=SUPPORTED_STRATEGIES,
        help=f"Ranking strategy (default: {DEFAULT_STRATEGY}).",
    )
    parser.add_argument(
        "-c", "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum calibrated confidence threshold (0.0 to 1.0).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw structured JSON instead of formatted text table.",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check and display asset health status.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Display model and asset configuration metadata.",
    )

    args = parser.parse_args()

    asset = NLPAsset(AssetConfig())

    if args.health:
        print(json.dumps(asset.health_check(), indent=2))
        return

    if args.info:
        info = {
            "metadata": asset.get_metadata().to_dict(),
            "supported_strategies": list(SUPPORTED_STRATEGIES),
            "health": asset.health_check(),
        }
        print(json.dumps(info, indent=2))
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    req = RecommendRequest(
        query=args.query,
        top_k=args.top_k,
        strategy=args.strategy,
        min_confidence=args.min_confidence,
    )

    response = asset.recommend(req)

    if args.json:
        print(response.to_json(indent=2))
    else:
        _print_table(response.to_dict())


if __name__ == "__main__":
    main()
