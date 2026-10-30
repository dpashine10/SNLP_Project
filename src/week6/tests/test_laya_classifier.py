"""Unit tests for the Laya System-1 Paper & Query Classification Engine."""

import unittest
from unittest.mock import MagicMock

from src.week6.laya_classifier import LayaPaperClassifier
from src.week6.schemas import ClassifyPaperInput


class TestLayaPaperClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = LayaPaperClassifier(prefer_neural=False, cache_size=128)

    def test_methodology_choice_distinguishes_five_paper_archetypes(self) -> None:
        papers = [
            ClassifyPaperInput(
                paper_id="W_SURV",
                title="A Comprehensive Survey and Literature Review of Graph Neural Networks",
                abstract="We present a systematic taxonomy of graph representation learning.",
            ),
            ClassifyPaperInput(
                paper_id="W_THEO",
                title="Minimax Convergence Bounds and Asymptotic Proofs for Stochastic Optimization",
                abstract="We prove tight upper and lower theoretical bounds under convexity.",
            ),
            ClassifyPaperInput(
                paper_id="W_RES",
                title="IndicBench: A Publicly Available Annotated Corpus and Benchmark Dataset",
                abstract="We release a gold-standard parallel corpus for shared task evaluation.",
            ),
            ClassifyPaperInput(
                paper_id="W_SYS",
                title="FPGA Hardware Accelerator and VLSI Architecture for Real-Time Inference",
                abstract="Embedded system prototype implemented on low-power ASIC hardware.",
            ),
            ClassifyPaperInput(
                paper_id="W_EMP",
                title="Experimental Fine-Tuning and Ablation Analysis for Sentiment Classification",
                abstract="We evaluate models across six benchmarks and report F1-score gains.",
            ),
        ]
        results = self.classifier.classify_papers(papers, query="machine learning")
        by_id = {r.paper_id: r for r in results}

        self.assertEqual(by_id["W_SURV"].paper_type, "survey")
        self.assertEqual(by_id["W_SURV"].paper_type_label, "Survey / Review")
        self.assertEqual(by_id["W_THEO"].paper_type, "theoretical")
        self.assertEqual(by_id["W_RES"].paper_type, "resource")
        self.assertEqual(by_id["W_SYS"].paper_type, "systems")
        self.assertEqual(by_id["W_EMP"].paper_type, "empirical")

        for item in results:
            self.assertAlmostEqual(sum(item.type_probabilities.values()), 1.0, places=2)
            self.assertGreater(item.type_confidence, 0.2)
            self.assertTrue(item.is_scholarly)

    def test_noul_flags_retractions_and_editorials_as_non_scholarly(self) -> None:
        papers = [
            ClassifyPaperInput(paper_id="W_RET", title="Retraction Note: Deep Neural Networks for NLP"),
            ClassifyPaperInput(paper_id="W_ED", title="Guest Editorial: Special Issue Welcome"),
            ClassifyPaperInput(
                paper_id="W_OK",
                title="Cross-Lingual Pretraining for Multilingual Question Answering",
                abstract="We evaluate transformer representations across twelve typologically diverse languages and demonstrate substantial zero-shot transfer improvements.",
            ),
        ]
        results = self.classifier.classify_papers(papers)
        by_id = {r.paper_id: r for r in results}
        self.assertFalse(by_id["W_RET"].is_scholarly)
        self.assertLess(by_id["W_RET"].scholarly_probability, 0.35)
        self.assertFalse(by_id["W_ED"].is_scholarly)
        self.assertTrue(by_id["W_OK"].is_scholarly)
        self.assertGreater(by_id["W_OK"].scholarly_probability, 0.70)

    def test_score_primitive_computes_graded_relevance_and_confidence_gate(self) -> None:
        papers = [
            ClassifyPaperInput(
                paper_id="W_HIGH",
                title="Transformer Language Models for Multilingual Machine Translation",
                abstract="Direct empirical evaluation of neural machine translation.",
                similarity=0.78,
            ),
            ClassifyPaperInput(
                paper_id="W_LOW",
                title="Soil Mechanics in Coastal Foundations",
                abstract="Geotechnical analysis of marine clay.",
                similarity=0.05,
            ),
        ]
        results = self.classifier.classify_papers(
            papers,
            query="multilingual machine translation",
            min_confidence=0.99,  # Strict threshold triggers Laya abstention gate
        )
        self.assertEqual(results[0].relevance_grade, 3)
        self.assertEqual(results[0].relevance_label, "Direct Match (3/3)")
        self.assertLessEqual(results[1].relevance_grade or 0, 1)
        self.assertTrue(results[1].abstained)

    def test_query_classification_identifies_domain_and_analytical_questions(self) -> None:
        analytical = self.classifier.classify_query(
            "Which researchers compare empirical vs theoretical transformer architectures for machine translation?"
        )
        self.assertEqual(analytical.primary_domain, "nlp")
        self.assertTrue(analytical.is_analytical_question)

        keyword = self.classifier.classify_query("quantum optics photonics")
        self.assertEqual(keyword.primary_domain, "quantum")
        self.assertFalse(keyword.is_analytical_question)

    def test_neural_agent_predict_batch_path_parses_laya_outputs(self) -> None:
        neural_clf = LayaPaperClassifier(prefer_neural=False)
        fake_agent = MagicMock()
        fake_agent.predict_batch.return_value = [
            {
                "model": "laya-typed-decisions",
                "answers": {
                    "paper_type": {
                        "type": "choice",
                        "choice": "survey",
                        "probabilities": {
                            "empirical": 0.08,
                            "theoretical": 0.04,
                            "survey": 0.81,
                            "resource": 0.04,
                            "systems": 0.03,
                        },
                        "confidence": 0.69,
                        "answer_confidence": 0.81,
                        "abstention": "passed",
                    },
                    "is_scholarly": {
                        "type": "noul",
                        "noul": 0.94,
                        "confidence": 0.94,
                        "answer_confidence": 0.94,
                    },
                    "relevance_grade": {
                        "type": "score",
                        "score": 2.76,
                        "probabilities": {"0": 0.01, "1": 0.04, "2": 0.13, "3": 0.82},
                        "confidence": 0.72,
                        "answer_confidence": 0.82,
                    },
                },
            }
        ]
        fake_agent.predict.return_value = {
            "model": "laya-typed-decisions",
            "answers": {
                "domain": {
                    "type": "choice",
                    "choice": "power",
                    "probabilities": {"nlp": 0.1, "ml": 0.05, "power": 0.75, "biomedical": 0.03, "systems": 0.04, "quantum": 0.03},
                    "answer_confidence": 0.75,
                },
                "is_analytical": {"type": "noul", "noul": 0.82},
            },
        }
        neural_clf._agent = fake_agent

        self.assertEqual(neural_clf.backend_name, "laya-typed-decisions")
        p_out = neural_clf.classify_papers(
            [ClassifyPaperInput(paper_id="W1", title="Survey of Power Grid Transformers")],
            query="How do power transformers diagnose insulation faults?",
        )
        self.assertEqual(p_out[0].backend, "laya-typed-decisions")
        self.assertEqual(p_out[0].paper_type, "survey")
        self.assertEqual(p_out[0].type_confidence, 0.81)
        self.assertEqual(p_out[0].relevance_grade, 3)
        self.assertTrue(p_out[0].is_scholarly)

        q_out = neural_clf.classify_query("How do power transformers diagnose insulation faults?")
        self.assertEqual(q_out.backend, "laya-typed-decisions")
        self.assertEqual(q_out.primary_domain, "power")
        self.assertTrue(q_out.is_analytical_question)


if __name__ == "__main__":
    unittest.main()

