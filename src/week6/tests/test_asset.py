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

    def test_recommendations_exclude_non_research_artifacts(self) -> None:
        from src.week6.filters import is_non_research_artifact

        res = self.asset.recommend(
            RecommendRequest(query="retraction note editorial code for ml models", top_k=5)
        )
        for rec in res.results:
            for ev in rec.evidence:
                self.assertFalse(is_non_research_artifact(ev.title), ev.title)

    def test_classify_endpoint_contract_and_convenience_helper(self) -> None:
        cls_res = self.asset.classify(
            {
                "query": "How do transformer models improve machine translation?",
                "papers": [
                    {
                        "paper_id": "W_SURVEY_1",
                        "title": "A Comprehensive Survey of Neural Machine Translation",
                        "abstract": "This review summarizes advances in transformer translation models.",
                        "similarity": 0.72,
                    },
                    {
                        "paper_id": "W_EMP_2",
                        "title": "Empirical Evaluation of Multilingual Transformer Adaptation",
                        "abstract": "Experiments across 12 language pairs demonstrate +3.4 BLEU improvement.",
                        "similarity": 0.68,
                    },
                ],
            }
        )
        self.assertEqual(len(cls_res.papers), 2)
        self.assertEqual(cls_res.papers[0].paper_type, "survey")
        self.assertIsNotNone(cls_res.query_classification)
        assert cls_res.query_classification is not None
        self.assertTrue(cls_res.query_classification.is_analytical_question)
        self.assertEqual(cls_res.query_classification.primary_domain, "nlp")
        self.assertEqual(cls_res.query_classification.primary_domain_label, "NLP & Language Models")

        direct_papers = self.asset.classify_papers(
            ["Benchmarking Vision-Language Retrieval Datasets"],
            query="multimodal retrieval benchmarks",
        )
        self.assertEqual(len(direct_papers), 1)
        self.assertEqual(direct_papers[0].paper_type, "resource")

    def test_opt_in_classify_papers_on_recommend(self) -> None:
        res_default = self.asset.recommend(
            RecommendRequest(query="natural language processing", top_k=2)
        )
        self.assertIsNone(res_default.query_classification)
        self.assertNotIn("query_classification", res_default.to_dict())
        if res_default.results and res_default.results[0].evidence:
            self.assertIsNone(res_default.results[0].evidence[0].classification)
            self.assertNotIn("classification", res_default.results[0].evidence[0].to_dict())

        res_classified = self.asset.recommend(
            RecommendRequest(query="natural language processing", top_k=2, classify_papers=True)
        )
        self.assertIsNotNone(res_classified.query_classification)
        self.assertIn("query_classification", res_classified.to_dict())
        for rec in res_classified.results:
            for ev in rec.evidence:
                self.assertIsNotNone(ev.classification)
                self.assertIn("classification", ev.to_dict())


if __name__ == "__main__":
    unittest.main()
