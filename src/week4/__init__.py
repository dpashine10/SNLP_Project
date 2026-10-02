"""Week 4: improved researcher recommendation using paper-level semantic search in ChromaDB.

This package is self-contained. Only ``data_adapter`` (Week 2 data) and
``baseline_adapter`` (Week 3 baseline) know about teammate code; every other
module communicates through the dataclasses in ``schemas``.

Entry points (run from the repository root):
    python -m src.week4.build_chromadb
    python -m src.week4.evaluation
"""
