"""Lightweight robustness check: how rankings change when the query is phrased differently.

Two kinds of query are used:

* Rephrasings of three benchmark queries (short keyword, detailed problem
  statement, abstract-style). They express the same information need as the
  base query, so each result is compared with the base query's ranking and,
  where a researcher was judged for the base query, with that judgment.
  Researchers never judged for the base query are counted as unjudged, not as
  irrelevant.
* Standalone ambiguous and Indian-domain queries. No labels exist for them,
  so only agreement between systems and the top researchers with their
  evidence are reported. No relevance is claimed for these.

The query texts below were written for this check; they are not part of the
Week 2 benchmark.
"""

import itertools
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.week5.judging.merge import Judgments
from src.week5.judging.schema import BINARY_RELEVANT_THRESHOLD

DEPTH = 10
TOP_LISTED = 5


@dataclass(frozen=True)
class RobustnessQuery:
    """One query of the check; ``base_query_id`` is set only for rephrasings of a benchmark query."""

    query_id: str
    kind: str
    text: str
    base_query_id: str | None = None


VARIANT_QUERIES: tuple[RobustnessQuery, ...] = (
    RobustnessQuery("q01-short", "short_keyword", "NLP", "q01"),
    RobustnessQuery(
        "q01-detailed", "detailed_problem",
        "We need a system that automatically understands and analyses large volumes of unstructured text, "
        "for tasks such as sentiment analysis, named entity recognition and question answering.",
        "q01",
    ),
    RobustnessQuery(
        "q01-abstract", "abstract_style",
        "In this paper, we propose a neural approach to natural language understanding. We train a language "
        "model on a large text corpus and evaluate it on parsing, machine translation and text classification "
        "benchmarks, where it outperforms strong baselines.",
        "q01",
    ),
    RobustnessQuery("q06-short", "short_keyword", "batteries", "q06"),
    RobustnessQuery(
        "q06-detailed", "detailed_problem",
        "Our project needs electrode materials that improve the capacity, charging rate and cycle life of "
        "lithium-ion and sodium-ion batteries for grid-scale energy storage.",
        "q06",
    ),
    RobustnessQuery(
        "q06-abstract", "abstract_style",
        "We report the synthesis and electrochemical characterization of a novel cathode material for "
        "rechargeable lithium-ion batteries. The material shows high specific capacity and excellent capacity "
        "retention over 500 charge-discharge cycles.",
        "q06",
    ),
    RobustnessQuery("q07-short", "short_keyword", "quantum", "q07"),
    RobustnessQuery(
        "q07-detailed", "detailed_problem",
        "We want to design quantum algorithms and error-correction schemes that run reliably on noisy "
        "intermediate-scale quantum hardware built from superconducting qubits.",
        "q07",
    ),
    RobustnessQuery(
        "q07-abstract", "abstract_style",
        "We present a scheme for fault-tolerant quantum computation using entangled qubits. Numerical "
        "simulations of the proposed quantum circuits show reduced gate errors compared with existing quantum "
        "error-correcting codes.",
        "q07",
    ),
)

STANDALONE_QUERIES: tuple[RobustnessQuery, ...] = (
    # "transformer": the neural architecture or the electrical power device.
    RobustnessQuery("amb-transformer", "ambiguous", "transformer"),
    # "network": computer, neural, social or power networks.
    RobustnessQuery("amb-network", "ambiguous", "network"),
    RobustnessQuery(
        "ind-crop", "indian_domain", "crop yield prediction for Indian agriculture using monsoon rainfall data"
    ),
    RobustnessQuery("ind-speech", "indian_domain", "speech recognition for Indian languages such as Hindi and Marathi"),
)


def jaccard(first: Sequence[str], second: Sequence[str]) -> float:
    """Overlap of two top-``DEPTH`` researcher sets; 1.0 when both are empty."""
    a, b = set(first[:DEPTH]), set(second[:DEPTH])
    return len(a & b) / len(a | b) if a | b else 1.0


def variant_rows(
    rankings: dict[tuple[str, str], list[str]], base_texts: dict[str, str], judgments: Judgments
) -> list[dict[str, Any]]:
    """Per system and rephrasing: overlap with the base ranking and judged precision.

    ``rankings`` maps (system, query_id) to ranked author IDs, where base
    queries use their benchmark ID (e.g. ``q01``). The base query itself is
    included as kind ``original`` so its judged precision is the reference.
    """
    systems = list(dict.fromkeys(system for system, _ in rankings))
    queries = [RobustnessQuery(base_id, "original", text, base_id) for base_id, text in base_texts.items()]
    queries += list(VARIANT_QUERIES)
    rows = []
    for system, query in itertools.product(systems, sorted(queries, key=lambda q: q.query_id)):
        base_id = query.base_query_id
        assert base_id is not None
        ranked = rankings[(system, query.query_id)][:DEPTH]
        base = rankings[(system, base_id)][:DEPTH]
        levels = [judgments[(base_id, author)] for author in ranked if (base_id, author) in judgments]
        relevant = sum(level >= BINARY_RELEVANT_THRESHOLD for level in levels)
        rows.append(
            {
                "system": system,
                "base_query_id": base_id,
                "base_query": base_texts[base_id],
                "kind": query.kind,
                "query_id": query.query_id,
                "query": query.text,
                "results": len(ranked),
                "shared_with_base": len(set(ranked) & set(base)),
                "jaccard_with_base": round(jaccard(ranked, base), 4),
                "same_top1": bool(ranked and base and ranked[0] == base[0]),
                "judged_results": len(levels),
                "relevant_results": relevant,
                "precision_among_judged": round(relevant / len(levels), 4) if levels else None,
            }
        )
    return rows


def variant_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per system and query kind: mean overlap with the base, judged coverage and judged precision."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["system"], row["kind"]), []).append(row)
    summary = []
    for (system, kind), group in groups.items():
        judged = sum(row["judged_results"] for row in group)
        relevant = sum(row["relevant_results"] for row in group)
        results = sum(row["results"] for row in group)
        summary.append(
            {
                "system": system,
                "kind": kind,
                "queries": len(group),
                "mean_jaccard_with_base": round(statistics.fmean(row["jaccard_with_base"] for row in group), 4),
                "same_top1": sum(row["same_top1"] for row in group),
                "results": results,
                "judged_coverage": round(judged / results, 4) if results else None,
                "precision_among_judged": round(relevant / judged, 4) if judged else None,
            }
        )
    return summary


def system_agreement(rankings: dict[tuple[str, str], list[str]], query_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Per query: mean pairwise top-``DEPTH`` Jaccard between systems, and Week 3 vs Week 4 separately."""
    systems = list(dict.fromkeys(system for system, _ in rankings))
    rows = []
    for query_id in query_ids:
        pairs = list(itertools.combinations(systems, 2))
        overlap = {pair: jaccard(rankings[(pair[0], query_id)], rankings[(pair[1], query_id)]) for pair in pairs}
        cross = [value for (a, b), value in overlap.items() if a.startswith("Week 3") != b.startswith("Week 3")]
        rows.append(
            {
                "query_id": query_id,
                "mean_pairwise_jaccard": round(statistics.fmean(overlap.values()), 4),
                "mean_week3_vs_week4_jaccard": round(statistics.fmean(cross), 4) if cross else None,
                "min_results": min(len(rankings[(system, query_id)]) for system in systems),
            }
        )
    return rows


def top_researchers(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The top ``TOP_LISTED`` researchers per system and standalone query, with evidence titles.

    Week 3 returns no paper evidence, so its researchers take the titles a
    Week 4 system returned for the same researcher and query, when one did.
    """
    titles: dict[tuple[str, str], str] = {}
    for record in records:
        for entry in record["ranking"]:
            if entry["evidence"]:
                titles.setdefault(
                    (record["query_id"], entry["author_id"]),
                    " || ".join(paper["title"] or "" for paper in entry["evidence"]),
                )
    return [
        {
            "query_id": record["query_id"],
            "kind": record["kind"],
            "query": record["query"],
            "system": record["system"],
            "rank": entry["rank"],
            "researcher_id": entry["author_id"],
            "researcher_name": entry["author_name"] or "",
            "evidence_paper_titles": titles.get((record["query_id"], entry["author_id"]), ""),
        }
        for record in records
        for entry in record["ranking"][:TOP_LISTED]
    ]
