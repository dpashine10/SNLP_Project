"""Unit and integration tests for the Flask REST API endpoints."""

import unittest
from unittest.mock import MagicMock

from src.week6.api import create_asset_app
from src.week6.schemas import (
    ConfidenceScore,
    ModelInfo,
    PaperEvidence,
    RecommendResponse,
    ResearcherRecommendation,
)


class TestAPI(unittest.TestCase):
    def setUp(self) -> None:
        # Mock NLP Asset to isolate API endpoint contract testing from model inference
        self.mock_asset = MagicMock()
        self.mock_asset.health_check.return_value = {
            "status": "ready",
            "ready": True,
            "backend": "fast_npz_paper_level",
            "model_name": "all-MiniLM-L6-v2",
            "default_strategy": "top_k_average_author_discount",
            "error": None,
        }
        self.mock_asset.get_metadata.return_value = ModelInfo(
            name="all-MiniLM-L6-v2",
            strategy="top_k_average_author_discount",
            backend="fast_npz_paper_level",
        )

        mock_conf = ConfidenceScore(
            score=0.88,
            level="HIGH",
            raw_score=0.55,
            margin=0.10,
            explanation="Strong contextual match.",
        )
        mock_evidence = PaperEvidence(
            paper_id="W999",
            title="Deep Learning Foundations",
            year="2024",
            similarity=0.82,
            authors_count=2,
        )
        mock_rec = ResearcherRecommendation(
            rank=1,
            author_id="A100",
            name="Yann LeCun",
            score=0.55,
            confidence=mock_conf,
            matched_papers=2,
            evidence=[mock_evidence],
            topics=["Computer Vision"],
        )
        self.mock_response = RecommendResponse(
            query="deep learning",
            execution_time_ms=15.2,
            total_results=1,
            model_info=self.mock_asset.get_metadata(),
            results=[mock_rec],
        )
        self.mock_asset.recommend.return_value = self.mock_response
        self.mock_asset.batch_recommend.return_value = [self.mock_response]

        self.app = create_asset_app(self.mock_asset)
        self.client = self.app.test_client()

    def test_health_endpoint(self) -> None:
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ready")
        self.assertTrue(data["ready"])

    def test_info_endpoint(self) -> None:
        res = self.client.get("/api/v1/info")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("model_metadata", data)
        self.assertIn("supported_strategies", data)
        self.assertIn("top_k_average_author_discount", data["supported_strategies"])

    def test_schema_endpoint(self) -> None:
        res = self.client.get("/api/v1/schema")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("RecommendRequest", data)
        self.assertIn("RecommendResponse", data)

    def test_recommend_endpoint_success(self) -> None:
        payload = {
            "query": "deep learning architectures",
            "top_k": 3,
            "strategy": "top_k_average_author_discount",
        }
        res = self.client.post("/api/v1/recommend", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["query"], "deep learning")
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["name"], "Yann LeCun")
        self.assertEqual(data["results"][0]["confidence"]["level"], "HIGH")

    def test_recommend_empty_query_validation_error(self) -> None:
        res = self.client.post("/api/v1/recommend", json={"query": "   "})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertEqual(data["error"], "Validation Error")

    def test_recommend_invalid_strategy_validation_error(self) -> None:
        res = self.client.post("/api/v1/recommend", json={"query": "nlp", "strategy": "invalid_mode"})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertEqual(data["error"], "Validation Error")

    def test_recommend_non_json_body(self) -> None:
        res = self.client.post("/api/v1/recommend", data="not json", content_type="text/plain")
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertEqual(data["error"], "Bad Request")

    def test_batch_endpoint(self) -> None:
        payload = {
            "queries": ["deep learning", "graph neural networks"],
            "top_k": 2,
        }
        res = self.client.post("/api/v1/batch", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["batch_size"], 1)
        self.assertIn("results", data)

    def test_not_found_endpoint(self) -> None:
        res = self.client.get("/api/v1/nonexistent")
        self.assertEqual(res.status_code, 404)

    def test_method_not_allowed(self) -> None:
        res = self.client.get("/api/v1/recommend")
        self.assertEqual(res.status_code, 405)


if __name__ == "__main__":
    unittest.main()
