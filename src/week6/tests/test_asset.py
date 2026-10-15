"""Integration tests for the NLPAsset class."""

import unittest

from src.week6.asset import AssetConfig, NLPAsset
from src.week6.schemas import RecommendRequest


class TestNLPAsset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Load asset once for the test case
        cls.asset = NLPAsset(AssetConfig())
        cls.asset.warmup()

    def test_warmup_and_health_check(self) -> None:
        health = self.asset.health_check()
        self.assertEqual(health["status"], "ready")
        self.assertTrue(health["ready"])
        self.assertIn("paper_level", health["backend"])

    def test_single_recommendation(self) -> None:
        req = RecommendRequest(
            query="neural machine translation and language models",
            top_k=3,
            strategy="top_k_average_author_discount",
            include_evidence=True,
        )
        res = self.asset.recommend(req)
        self.assertEqual(res.query, req.query)
        self.assertEqual(len(res.results), 3)
        self.assertGreater(res.execution_time_ms, 0)
        self.assertEqual(res.model_info.strategy, "top_k_average_author_discount")

        for r in res.results:
            self.assertTrue(r.name)
            self.assertTrue(r.author_id)
            self.assertGreater(r.score, 0)
            self.assertIn(r.confidence.level, ["HIGH", "MEDIUM", "LOW"])
            self.assertGreaterEqual(r.confidence.score, 0.0)
            self.assertLessEqual(r.confidence.score, 1.0)
            if r.evidence:
                self.assertTrue(r.evidence[0].title)

    def test_strategy_switch_best_paper(self) -> None:
        req = RecommendRequest(
            query="quantum computing algorithms",
            top_k=2,
            strategy="best_paper",
        )
        res = self.asset.recommend(req)
        self.assertEqual(res.model_info.strategy, "best_paper")
        self.assertEqual(len(res.results), 2)

    def test_min_confidence_filter(self) -> None:
        req = RecommendRequest(
            query="information retrieval",
            top_k=5,
            min_confidence=0.999,  # Impossibly high to ensure filtering works
        )
        res = self.asset.recommend(req)
        self.assertEqual(len(res.results), 0)

    def test_batch_recommendation(self) -> None:
        queries = [
            "deep reinforcement learning",
            "computer vision and convolutional neural networks",
        ]
        responses = self.asset.batch_recommend(queries)
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0].query, queries[0])
        self.assertEqual(responses[1].query, queries[1])


if __name__ == "__main__":
    unittest.main()
