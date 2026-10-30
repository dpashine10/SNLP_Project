"""Document-level quality filters for excluding non-research records.

Addresses documented Week 5 error analysis failure modes:
- F2: Bare code-deposit artifacts (e.g., "Code for the ML models")
- F3: Retraction notices repeating original paper titles ("Retraction Note to: ...")
- F4: Standalone editorials, errata, and publisher corrections ("Editorial")
"""

from __future__ import annotations

_EXACT_NON_RESEARCH_TITLES = frozenset({
    "editorial",
    "guest editorial",
    "editorial board",
    "preface",
    "foreword",
    "erratum",
    "corrigendum",
    "retraction",
    "retraction note",
    "acknowledgment",
    "acknowledgement",
    "index",
})

_NON_RESEARCH_PREFIXES = (
    "retraction note",
    "retraction:",
    "retracted:",
    "retracted article",
    "withdrawal:",
    "withdrawn:",
    "erratum:",
    "erratum to:",
    "erratum to ",
    "corrigendum:",
    "corrigendum to:",
    "corrigendum to ",
    "publisher correction:",
    "author correction:",
    "editorial:",
    "guest editorial:",
    "code for ",
    "supplementary code for ",
    "replication code for ",
)


def is_non_research_artifact(title: str | None) -> bool:
    """Return True when ``title`` identifies a notice, editorial, or code stub rather than a paper."""
    if not title:
        return True
    normalized = " ".join(str(title).strip().lower().split())
    if not normalized or normalized == "unknown title":
        return True
    if normalized in _EXACT_NON_RESEARCH_TITLES:
        return True
    return normalized.startswith(_NON_RESEARCH_PREFIXES)

