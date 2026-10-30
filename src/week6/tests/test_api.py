"""Unit and integration tests for the Flask REST API endpoints."""

import unittest
from unittest.mock import MagicMock

from src.week6.api import create_asset_app
from src.week6.schemas import (
    CitationSource,
    ClassifyResponse,
    ConfidenceScore,
    ModelInfo,
    PaperClassification,
    PaperEvidence,
    QueryClassification,
    ReasoningResponse,
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
        self.assertIn("ranking_explanation", data["model_metadata"])
        self.assertEqual(data["university_integration"]["consumer"], "group6_for_group7 integration bridge")
        self.assertTrue(data["university_integration"]["advisory_only"])

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

    def test_batch_validation_returns_400_instead_of_500(self) -> None:
        for payload in (
            {"queries": ["nlp"], "top_k": "not-an-int"},
            {"queries": ["nlp"], "strategy": False},
            {"queries": [{"query": "nlp", "include_evidence": "false"}]},
        ):
            res = self.client.post("/api/v1/batch", json=payload)
            self.assertEqual(res.status_code, 400, payload)
            self.assertEqual(res.get_json()["error"], "Validation Error")

    def test_not_found_endpoint(self) -> None:
        res = self.client.get("/api/v1/nonexistent")
        self.assertEqual(res.status_code, 404)

    def test_method_not_allowed(self) -> None:
        res = self.client.get("/api/v1/recommend")
        self.assertEqual(res.status_code, 405)

    def test_background_warmup_option_invokes_start_warmup(self) -> None:
        mock_asset = MagicMock()
        create_asset_app(mock_asset, background_warmup=True)
        mock_asset.start_warmup.assert_called_once_with()

    def test_classify_endpoint_success_and_validation(self) -> None:
        self.mock_asset.classify.return_value = ClassifyResponse(
            execution_time_ms=4.5,
            backend="laya-calibrated-local",
            papers=[
                PaperClassification(
                    paper_id="W999",
                    title="A Survey of Neural Machine Translation",
                    paper_type="survey",
                    paper_type_label="Survey / Review",
                    type_confidence=0.84,
                    type_probabilities={"survey": 0.84, "empirical": 0.16},
                    is_scholarly=True,
                    scholarly_probability=0.93,
                    relevance_score=0.92,
                    relevance_grade=3,
                    relevance_label="Direct Match (3/3)",
                    backend="laya-calibrated-local",
                )
            ],
            query_classification=QueryClassification(
                query="How do transformer surveys evaluate translation?",
                primary_domain="nlp_language_models",
                primary_domain_label="NLP & Language Models",
                domain_confidence=0.81,
                domain_probabilities={"nlp_language_models": 0.81},
                is_analytical_question=True,
                analytical_probability=0.88,
                backend="laya-calibrated-local",
            ),
        )
        res = self.client.post(
            "/api/v1/classify",
            json={
                "query": "How do transformer surveys evaluate translation?",
                "papers": [{"paper_id": "W999", "title": "A Survey of Neural Machine Translation"}],
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["backend"], "laya-calibrated-local")
        self.assertEqual(len(data["papers"]), 1)
        self.assertEqual(data["papers"][0]["paper_type"], "survey")
        self.assertEqual(data["papers"][0]["relevance_grade"], 3)
        self.assertEqual(data["query_classification"]["primary_domain"], "nlp_language_models")

        # Empty payload without papers or query -> 400 Validation Error
        bad_empty = self.client.post("/api/v1/classify", json={})
        self.assertEqual(bad_empty.status_code, 400)
        self.assertEqual(bad_empty.get_json()["error"], "Validation Error")

        # Out-of-range min_confidence -> 400 Validation Error
        bad_conf = self.client.post("/api/v1/classify", json={"query": "nlp", "min_confidence": 1.5})
        self.assertEqual(bad_conf.status_code, 400)
        self.assertEqual(bad_conf.get_json()["error"], "Validation Error")

        # Non-JSON body -> 400 Bad Request
        bad_body = self.client.post("/api/v1/classify", data="plain text", content_type="text/plain")
        self.assertEqual(bad_body.status_code, 400)
        self.assertEqual(bad_body.get_json()["error"], "Bad Request")

    def test_info_and_schema_expose_laya_classification_contracts(self) -> None:
        info_data = self.client.get("/api/v1/info").get_json()
        self.assertIn("laya_classifier", info_data)
        self.assertIn("reasoner", info_data)
        self.assertIn("POST /api/v1/classify", info_data["available_endpoints"])
        self.assertIn("POST /api/v1/reason", info_data["available_endpoints"])

        schema_data = self.client.get("/api/v1/schema").get_json()
        self.assertIn("ClassifyRequest", schema_data)
        self.assertIn("ClassifyResponse", schema_data)
        self.assertIn("ReasoningRequest", schema_data)
        self.assertIn("ReasoningResponse", schema_data)
        self.assertIn("classify_papers", schema_data["RecommendRequest"]["properties"])
        self.assertIn("include_reasoning", schema_data["RecommendRequest"]["properties"])

    def test_reason_endpoint_success_and_validation(self) -> None:
        self.mock_asset.reason.return_value = ReasoningResponse(
            query="transformers in nlp",
            answer="Transformers use multi-head attention mechanisms [Paper #1 · Smith, 2024].",
            provider_used="groq",
            model_name="llama-3.3-70b-versatile",
            citations=[
                CitationSource(
                    paper_index=1,
                    title="Transformer Models in Natural Language Processing",
                    paper_id="W1",
                    year="2024",
                )
            ],
            execution_time_ms=18.4,
            is_fallback=False,
        )
        res = self.client.post(
            "/api/v1/reason",
            json={
                "query": "transformers in nlp",
                "papers": [{"title": "Transformer Models in Natural Language Processing"}],
                "provider": "auto",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["provider_used"], "groq")
        self.assertIn("multi-head attention", data["answer"])
        self.assertEqual(len(data["citations"]), 1)

        # Empty query -> 400 Validation Error
        bad_empty = self.client.post("/api/v1/reason", json={"query": "   ", "papers": []})
        self.assertEqual(bad_empty.status_code, 400)
        self.assertEqual(bad_empty.get_json()["error"], "Validation Error")

        # Invalid provider -> 400 Validation Error
        bad_prov = self.client.post("/api/v1/reason", json={"query": "nlp", "provider": "invalid_llm"})
        self.assertEqual(bad_prov.status_code, 400)
        self.assertEqual(bad_prov.get_json()["error"], "Validation Error")

        # Non-JSON body -> 400 Bad Request
        bad_body = self.client.post("/api/v1/reason", data="plain text", content_type="text/plain")
        self.assertEqual(bad_body.status_code, 400)
        self.assertEqual(bad_body.get_json()["error"], "Bad Request")


if __name__ == "__main__":
    unittest.main()
