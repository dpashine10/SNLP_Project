"""Cross-repository HTTP contract test for the Group 6 bridge.

The test uses a mocked NLP asset so it verifies the real Flask serializer and
HTTP route without loading the embedding model. The Group 6 bridge parser then
consumes the actual JSON response shape.
"""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

from werkzeug.serving import make_server

from src.week6.api import create_asset_app
from src.week6.schemas import (
    ConfidenceScore,
    ModelInfo,
    PaperEvidence,
    RecommendResponse,
    ResearcherRecommendation,
)

GROUP6_INTEGRATION = Path(__file__).resolve().parents[1].parent / "group6_for_group7" / "integration"
if str(GROUP6_INTEGRATION) not in sys.path:
    sys.path.insert(0, str(GROUP6_INTEGRATION))

from s3t_to_discovery import call_group7, parse_mentors  # noqa: E402


class TestGroup6BridgeContract(unittest.TestCase):
    def test_real_flask_api_response_is_consumable_by_group6_bridge(self) -> None:
        class FakeAsset:
            def recommend(self, request):  # type: ignore[no-untyped-def]
                return RecommendResponse(
                    query=request.query,
                    execution_time_ms=1.2,
                    total_results=1,
                    model_info=ModelInfo(
                        name="all-MiniLM-L6-v2",
                        strategy=request.strategy,
                        backend="test",
                    ),
                    results=[
                        ResearcherRecommendation(
                            rank=1,
                            author_id="A1",
                            name="Researcher Example",
                            score=0.61,
                            confidence=ConfidenceScore(
                                score=0.82,
                                level="HIGH",
                                raw_score=0.61,
                                margin=0.04,
                                explanation="Evidence-backed test match.",
                            ),
                            matched_papers=1,
                            evidence=[
                                PaperEvidence(
                                    paper_id="W1",
                                    title="Evidence paper",
                                    year="2024",
                                    similarity=0.74,
                                    authors_count=2,
                                )
                            ],
                        )
                    ],
                )

        app = create_asset_app(FakeAsset())
        server = make_server("127.0.0.1", 0, app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"
            response = call_group7("natural language processing", base_url, timeout=2.0)
            mentors = parse_mentors(response)
        finally:
            server.shutdown()
            thread.join(timeout=2)

        self.assertEqual(response["results"][0]["name"], "Researcher Example")
        self.assertEqual(mentors[0].top_papers[0]["title"], "Evidence paper")
        self.assertEqual(mentors[0].confidence_level, "HIGH")


if __name__ == "__main__":
    unittest.main()
