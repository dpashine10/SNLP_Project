"""Agreement between the old TF-IDF-derived labels and the new manual relevance labels.

The old benchmark labels 80 pooled pairs explicitly (``existing_label = 1``).
Its metrics treated every other returned researcher as not relevant, so the
remaining 134 pairs are *implied* negatives. A new label of 2 or 3 counts as
relevant. Agreement is reported for the explicit labels, the implied
negatives and both together.
"""

from collections import Counter
from typing import Any

import pandas as pd

from src.week5.judging.schema import BINARY_RELEVANT_THRESHOLD, FLAGGED_VALUES

_SYSTEM_PATTERN = r"(Week \d \([a-z_]+\)) #\d+"


def label_frame(judgments: pd.DataFrame) -> pd.DataFrame:
    """Add binary old/new relevance and agreement columns to fully judged rows."""
    frame = judgments.copy()
    frame["level"] = frame["manual_relevance"].astype(int)
    frame["old_relevant"] = frame["existing_label"] == "1"
    frame["new_relevant"] = frame["level"] >= BINARY_RELEVANT_THRESHOLD
    frame["agree"] = frame["old_relevant"] == frame["new_relevant"]
    frame["flagged"] = frame["review_flag"].str.strip().isin(FLAGGED_VALUES)
    return frame


def agreement_report(judgments: pd.DataFrame) -> dict[str, Any]:
    """Overall, per-query, per-system and flag analyses of old vs new labels."""
    frame = label_frame(judgments)
    explicit, implied = frame[frame["old_relevant"]], frame[~frame["old_relevant"]]
    return {
        "label_provenance": {
            "judge_ids": dict(Counter(frame["judge_id"])),
            "distinct_judged_at": int(frame["judged_at"].nunique()),
        },
        "overall": _agreement(frame),
        "explicit_old_labels": _agreement(explicit),
        "implied_old_negatives": _agreement(implied),
        "confusion": {
            "old_label_1": _levels(explicit),
            "old_label_blank": _levels(implied),
        },
        "by_query": {
            query_id: {"query": group["query_text"].iloc[0], **_agreement(group)}
            for query_id, group in frame.groupby("query_id")
        },
        "by_system": _by_system(frame),
        "flagged": _flag_analysis(frame),
    }


def disagreements(judgments: pd.DataFrame) -> pd.DataFrame:
    """Rows where the old and new binary labels differ."""
    frame = label_frame(judgments)
    columns = [
        "query_id", "query_text", "researcher_id", "researcher_name", "system_source",
        "existing_label", "manual_relevance", "review_flag", "judgment_notes",
    ]
    return frame.loc[~frame["agree"], columns]


def _agreement(frame: pd.DataFrame) -> dict[str, Any]:
    total, agree = len(frame), int(frame["agree"].sum())
    return {
        "rows": total,
        "match": agree,
        "differ": total - agree,
        "disagreement_rate": round((total - agree) / total, 4) if total else None,
        "old_relevant": int(frame["old_relevant"].sum()),
        "new_relevant": int(frame["new_relevant"].sum()),
    }


def _levels(frame: pd.DataFrame) -> dict[str, int]:
    counts = Counter(frame["level"])
    return {str(level): counts.get(level, 0) for level in range(4)}


def _by_system(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Each pair counted once for every system that returned it."""
    exploded = frame.assign(system=frame["system_source"].str.findall(_SYSTEM_PATTERN)).explode("system")
    order = list(dict.fromkeys(exploded["system"]))
    return {system: _agreement(exploded[exploded["system"] == system]) for system in sorted(order)}


def _flag_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    flagged = frame[frame["flagged"]]
    no_evidence = flagged["evidence_paper_ids"] == ""
    return {
        "flagged_rows": len(flagged),
        "by_query": {query_id: len(group) for query_id, group in flagged.groupby("query_id")},
        "reasons": {
            "no Week 4 evidence; judged from profile titles only": int(no_evidence.sum()),
            "Week 4 evidence present but weak or non-substantive": int((~no_evidence).sum()),
        },
        "weak_evidence_rows": [
            {
                "query_id": row.query_id,
                "researcher_name": row.researcher_name,
                "manual_relevance": row.manual_relevance,
                "evidence_paper_titles": row.evidence_paper_titles,
                "judgment_notes": row.judgment_notes,
            }
            for row in flagged[~no_evidence].itertuples(index=False)
        ],
        "levels_flagged": _levels(flagged),
        "levels_not_flagged": _levels(frame[~frame["flagged"]]),
    }


def report_markdown(report: dict[str, Any]) -> str:
    """Short Markdown version of the agreement report."""
    provenance = report["label_provenance"]
    lines = [
        "# Old vs new labels (Week 5, Phase 2)",
        "",
        "Old label: `existing_label` (TF-IDF-derived; blank was scored as not relevant). "
        "New label: `manual_relevance` from `judging/manual_judgments_completed.csv`; levels 2–3 count as relevant.",
        "",
        f"Label provenance: judge_id {provenance['judge_ids']}, {provenance['distinct_judged_at']} distinct "
        "`judged_at` value(s). These labels were produced by an AI reviewer, not a human judge.",
        "",
        "| Slice | Rows | Match | Differ | Disagreement rate | Old relevant | New relevant |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, key in (
        ("All pooled pairs", "overall"),
        ("Explicit old labels (existing_label = 1)", "explicit_old_labels"),
        ("Implied old negatives (existing_label blank)", "implied_old_negatives"),
    ):
        lines.append(_row(name, report[key]))
    confusion = report["confusion"]
    lines += [
        "",
        "New level by old label:",
        "",
        "| Old label | 0 | 1 | 2 | 3 |",
        "|---|---|---|---|---|",
        "| 1 | " + " | ".join(str(v) for v in confusion["old_label_1"].values()) + " |",
        "| blank | " + " | ".join(str(v) for v in confusion["old_label_blank"].values()) + " |",
        "",
        "## By query",
        "",
        "| Query | Rows | Match | Differ | Disagreement rate | Old relevant | New relevant |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [_row(f"{query_id} {row['query']}", row) for query_id, row in report["by_query"].items()]
    lines += [
        "",
        "## By system (each pair counted for every system that returned it)",
        "",
        "| System | Rows | Match | Differ | Disagreement rate | Old relevant | New relevant |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [_row(system, row) for system, row in report["by_system"].items()]
    flagged = report["flagged"]
    lines += [
        "",
        "## Rows flagged for review",
        "",
        f"{flagged['flagged_rows']} rows have `review_flag = yes`. Reasons (from the evidence columns):",
        "",
    ]
    lines += [f"- {reason}: {count}" for reason, count in flagged["reasons"].items()]
    lines += ["", "By query: " + ", ".join(f"{q}: {n}" for q, n in flagged["by_query"].items()), ""]
    lines += [
        f"Levels when flagged: {flagged['levels_flagged']}; when not flagged: {flagged['levels_not_flagged']}.",
        "",
        "Weak-evidence rows:",
        "",
    ]
    lines += [
        f"- {row['query_id']} {row['researcher_name']} (level {row['manual_relevance']}): {row['evidence_paper_titles'][:90]}"
        for row in flagged["weak_evidence_rows"]
    ]
    return "\n".join(lines) + "\n"


def _row(name: str, row: dict[str, Any]) -> str:
    return (
        f"| {name} | {row['rows']} | {row['match']} | {row['differ']} | {row['disagreement_rate']:.1%} | "
        f"{row['old_relevant']} | {row['new_relevant']} |"
    )
