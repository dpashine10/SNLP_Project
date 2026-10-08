"""Structured error analysis over the frozen rankings.

Only failures visible in the frozen evidence itself are recorded as confirmed
cases. The structural Week 4 problem of many co-authors from one paper
sharing a score is visible without labels. Anything that depends on whether a
researcher is relevant goes to a review queue until manual judgments exist.
No case is invented and Week 4's ranking code is not changed.
"""

from collections import Counter
from collections.abc import Sequence
from typing import Any

from src.week5.judging.merge import Judgments

# A best-evidence paper shared by at least this many top-10 researchers is flagged.
SHARED_PAPER_MIN = 3
# At least this many top-10 researchers with exactly the same score is flagged.
TIE_GROUP_MIN = 3
# A top 10 drawn from this many distinct best papers or fewer is flagged.
FEW_PAPERS_MAX = 5
# Review-queue depth: researchers ranked this high whose relevance decides errors.
REVIEW_DEPTH = 5
REVIEW_SYSTEMS = ("Week 3 (semantic)", "Week 4 (best_paper)", "Week 4 (average)", "Week 4 (top_k_average)")

ERROR_CASE_COLUMNS = (
    "case_id", "query_id", "query_text", "system", "researcher_name", "rank", "score", "relevance_label",
    "evidence_paper_title", "error_category", "observed_failure", "likely_cause", "impact",
    "possible_future_fix", "confidence", "reviewer_notes",
)


def coauthor_tie_report(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per Week 4 system and query: how concentrated the top 10 is on few papers and tied scores."""
    rows = []
    for record in records:
        if record["family"] != "week4":
            continue
        ranking = record["ranking"]
        best_papers = Counter(_best_paper_id(entry) for entry in ranking)
        scores = Counter(entry["score"] for entry in ranking)
        shared_id, shared_count = best_papers.most_common(1)[0] if best_papers else ("", 0)
        largest_tie = max(scores.values(), default=0)
        rows.append(
            {
                "system": record["system"],
                "query_id": record["query_id"],
                "query": record["query"],
                "researchers": len(ranking),
                "distinct_best_papers": len(best_papers),
                "largest_shared_paper_group": shared_count,
                "largest_shared_paper_title": _title_of(ranking, shared_id),
                "largest_identical_score_group": largest_tie,
                "tied_at_top_score": sum(1 for entry in ranking if ranking and entry["score"] == ranking[0]["score"]),
                "flag_shared_paper": shared_count >= SHARED_PAPER_MIN,
                "flag_identical_scores": largest_tie >= TIE_GROUP_MIN,
                "flag_few_papers": len(best_papers) <= FEW_PAPERS_MAX,
            }
        )
    return rows


def confirmed_cases(records: Sequence[dict[str, Any]], judgments: Judgments) -> list[dict[str, Any]]:
    """Error cases supported directly by the frozen evidence (no relevance assumption)."""
    cases: list[dict[str, Any]] = []
    for record in records:
        if record["family"] != "week4":
            continue
        groups: dict[str, list[dict[str, Any]]] = {}
        for entry in record["ranking"]:
            groups.setdefault(_best_paper_id(entry), []).append(entry)
        for paper_id, entries in groups.items():
            if len(entries) < SHARED_PAPER_MIN:
                continue
            cases.append(_shared_paper_case(len(cases) + 1, record, paper_id, entries, judgments))
    return cases


def review_queue(records: Sequence[dict[str, Any]], judgments: Judgments) -> list[dict[str, Any]]:
    """Top-ranked researchers whose manual relevance decides which errors exist."""
    queue: dict[tuple[str, str], dict[str, Any]] = {}
    shared = {
        (record["query_id"], entry["author_id"])
        for record in records
        if record["family"] == "week4"
        for paper_id, count in Counter(_best_paper_id(e) for e in record["ranking"]).items()
        if count >= SHARED_PAPER_MIN
        for entry in record["ranking"]
        if _best_paper_id(entry) == paper_id
    }
    for record in records:
        if record["system"] not in REVIEW_SYSTEMS:
            continue
        for entry in record["ranking"][:REVIEW_DEPTH]:
            key = (record["query_id"], entry["author_id"])
            item = queue.setdefault(
                key,
                {
                    "query_id": record["query_id"],
                    "query_text": record["query"],
                    "researcher_id": entry["author_id"],
                    "researcher_name": entry["author_name"] or "",
                    "returned_at": [],
                    "best_rank": entry["rank"],
                    "existing_label": "1" if entry["labelled_relevant"] else "",
                    "manual_relevance": judgments.get(key, ""),
                    "top_evidence_paper": _title_of(record["ranking"], _best_paper_id(entry)) if entry["evidence"] else "",
                    "in_shared_paper_group": key in shared,
                },
            )
            item["returned_at"].append(f"{record['system']} #{entry['rank']}")
            item["best_rank"] = min(item["best_rank"], entry["rank"])
            if not item["top_evidence_paper"] and entry["evidence"]:
                item["top_evidence_paper"] = _title_of(record["ranking"], _best_paper_id(entry))
    rows = [
        {**item, "returned_at": "; ".join(item["returned_at"]), "pending_check": _pending_check(item)}
        for item in queue.values()
    ]
    return sorted(rows, key=lambda row: (row["query_id"], row["best_rank"], row["researcher_id"]))


def _pending_check(item: dict[str, Any]) -> str:
    if item["manual_relevance"] == "":
        return "Judge relevance first; a top-5 result judged 0 or 1 becomes an error case to categorise."
    if int(item["manual_relevance"]) <= 1:
        return "Judged 0/1 while ranked in the top 5: categorise the error from the evidence."
    return "No error: judged relevant."


def _shared_paper_case(
    number: int, record: dict[str, Any], paper_id: str, entries: list[dict[str, Any]], judgments: Judgments
) -> dict[str, Any]:
    scores = {entry["score"] for entry in entries}
    identical = len(scores) == 1
    labels = [judgments.get((record["query_id"], entry["author_id"])) for entry in entries]
    relevance = "unjudged" if any(label is None for label in labels) else "; ".join(str(label) for label in labels)
    ranks = [entry["rank"] for entry in entries]
    return {
        "case_id": f"E{number:03d}",
        "query_id": record["query_id"],
        "query_text": record["query"],
        "system": record["system"],
        "researcher_name": "; ".join(entry["author_name"] or entry["author_id"] for entry in entries),
        "rank": "; ".join(str(rank) for rank in ranks),
        "score": f"{min(scores):.4f}" if identical else f"{min(scores):.4f}–{max(scores):.4f}",
        "relevance_label": relevance,
        "evidence_paper_title": _title_of(record["ranking"], paper_id),
        "error_category": "co-author tie inflation" if identical else "too many researchers from one paper",
        "observed_failure": (
            f"{len(entries)} of the top {len(record['ranking'])} researchers (ranks {ranks[0]}–{ranks[-1]}) share "
            f"the same best evidence paper"
            + (f" and the identical score {min(scores):.4f}; their order is set only by author ID." if identical else ".")
        ),
        "likely_cause": (
            "Each co-author of a matching paper is credited with that paper's full similarity score; "
            "the strategy has no co-author or multi-paper adjustment."
        ),
        "impact": "Fewer distinct papers and research groups in the top 10; positions inside the tie are arbitrary.",
        "possible_future_fix": "Co-author-aware scoring or tie-breaking (e.g. favour researchers with several matching papers).",
        "confidence": "high (directly observed in the frozen rankings)",
        "reviewer_notes": (
            "Structural observation only; whether it lowers accuracy depends on the manual relevance labels."
            if relevance == "unjudged"
            else "Manual labels available; check whether the tied researchers are equally relevant."
        ),
    }


def _best_paper_id(entry: dict[str, Any]) -> str:
    evidence = entry["evidence"]
    return str(evidence[0]["paper_id"]) if evidence else f"(no evidence: {entry['author_id']})"


def _title_of(ranking: Sequence[dict[str, Any]], paper_id: str) -> str:
    for entry in ranking:
        for paper in entry["evidence"]:
            if paper["paper_id"] == paper_id:
                return str(paper["title"] or "(untitled)")
    return ""
