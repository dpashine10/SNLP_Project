"""Unit tests for Week 6 structured input and output schemas."""

import unittest

from src.week6.schemas import (
    DEFAULT_PAPER_TOP_K,
    DEFAULT_STRATEGY,
    DEFAULT_TOP_K,
    ConfidenceScore,
    ModelInfo,
    PaperEvidence,
    RecommendRequest,
    RecommendResponse,
    ResearcherRecommendation,
    ValidationError,
)


class TestSchemas(unittest.TestCase):
    def test_valid_recommend_request(self) -> None:
        req = RecommendRequest(
            query="machine learning",
            top_k=10,
            strategy="best_paper",
            min_confidence=0.5,
            include_evidence=True,
            paper_top_k=80,
        )
        self.assertEqual(req.query, "machine learning")
        self.assertEqual(req.top_k, 10)
        self.assertEqual(req.strategy, "best_paper")
        self.assertEqual(req.min_confidence, 0.5)
        self.assertTrue(req.include_evidence)
        self.assertEqual(req.paper_top_k, 80)

    def test_default_values(self) -> None:
        req = RecommendRequest(query="deep neural networks")
        self.assertEqual(req.top_k, DEFAULT_TOP_K)
        self.assertEqual(req.strategy, DEFAULT_STRATEGY)
        self.assertEqual(req.min_confidence, 0.0)
        self.assertTrue(req.include_evidence)
        self.assertEqual(req.paper_top_k, DEFAULT_PAPER_TOP_K)

    def test_empty_query_raises_validation_error(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RecommendRequest(query="   ")
        self.assertEqual(ctx.exception.field_name, "query")

    def test_invalid_top_k_raises_validation_error(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RecommendRequest(query="nlp", top_k=0)
        self.assertEqual(ctx.exception.field_name, "top_k")

        with self.assertRaises(ValidationError) as ctx:
            RecommendRequest(query="nlp", top_k=100)
        self.assertEqual(ctx.exception.field_name, "top_k")

    def test_invalid_strategy_raises_validation_error(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RecommendRequest(query="nlp", strategy="unsupported_strategy")
        self.assertEqual(ctx.exception.field_name, "strategy")

    def test_invalid_min_confidence_raises_validation_error(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RecommendRequest(query="nlp", min_confidence=-0.1)
        self.assertEqual(ctx.exception.field_name, "min_confidence")

        with self.assertRaises(ValidationError) as ctx:
            RecommendRequest(query="nlp", min_confidence=1.5)
        self.assertEqual(ctx.exception.field_name, "min_confidence")

    def test_from_dict_and_to_dict(self) -> None:
        payload = {
            "query": "transformer attention",
            "top_k": 3,
            "strategy": "top_k_average",
            "min_confidence": 0.4,
            "include_evidence": False,
            "paper_top_k": 40,
        }
        req = RecommendRequest.from_dict(payload)
        self.assertEqual(req.query, "transformer attention")
        self.assertEqual(req.top_k, 3)
        self.assertEqual(req.strategy, "top_k_average")
        self.assertFalse(req.include_evidence)

        data = req.to_dict()
        self.assertEqual(data["query"], "transformer attention")
        self.assertEqual(data["top_k"], 3)

    def test_from_dict_rejects_ambiguous_json_types(self) -> None:
        with self.assertRaises(ValidationError):
            RecommendRequest.from_dict({"query": "nlp", "include_evidence": "false"})
        with self.assertRaises(ValidationError):
            RecommendRequest.from_dict({"query": "nlp", "top_k": "5"})

    def test_paper_top_k_has_documented_upper_bound(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RecommendRequest(query="nlp", paper_top_k=201)
        self.assertEqual(ctx.exception.field_name, "paper_top_k")

    def test_recommend_response_serialization(self) -> None:
        conf = ConfidenceScore(
            score=0.85,
            level="HIGH",
            raw_score=0.62,
            margin=0.08,
            explanation="Strong contextual match.",
        )
        evidence = PaperEvidence(
            paper_id="W123",
            title="Attention Is All You Need",
            year="2017",
            similarity=0.88,
            authors_count=8,
        )
        rec = ResearcherRecommendation(
            rank=1,
            author_id="A456",
            name="Ashish Vaswani",
            score=0.62,
            confidence=conf,
            matched_papers=3,
            evidence=[evidence],
            topics=["NLP", "Transformers"],
        )
        model_info = ModelInfo(
            name="all-MiniLM-L6-v2",
            strategy="top_k_average_author_discount",
            backend="fast_npz_paper_level",
        )
        res = RecommendResponse(
            query="transformers",
            execution_time_ms=12.4,
            total_results=1,
            model_info=model_info,
            results=[rec],
        )

        d = res.to_dict()
        self.assertEqual(d["query"], "transformers")
        self.assertEqual(d["total_results"], 1)
        self.assertEqual(d["results"][0]["name"], "Ashish Vaswani")
        self.assertEqual(d["results"][0]["confidence"]["level"], "HIGH")
        self.assertEqual(d["results"][0]["evidence"][0]["title"], "Attention Is All You Need")

        json_str = res.to_json()
        self.assertIn("Ashish Vaswani", json_str)
        self.assertIn("HIGH", json_str)


if __name__ == "__main__":
    unittest.main()
