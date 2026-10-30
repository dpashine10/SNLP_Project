"""Unit tests for the Grounded Reasoning Engine (Groq / Gemini / Extractive Fallback)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.week6.reasoning import GroundedReasoningEngine
from src.week6.schemas import CitationSource, ReasoningRequest, ReasoningResponse


class TestGroundedReasoningEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.papers = [
            CitationSource(
                paper_index=1,
                title="Transformer Models in Natural Language Processing",
                paper_id="https://openalex.org/W1",
                year="2024",
                abstract="We investigate multi-head attention mechanisms and scaling laws in transformer architectures.",
                doi="10.1145/3600000",
                authors=("Alice Smith", "Bob Jones"),
                similarity=0.88,
            ),
            CitationSource(
                paper_index=2,
                title="Empirical Evaluation of Multilingual Translation",
                paper_id="https://openalex.org/W2",
                year="2023",
                abstract="Cross-lingual evaluation across 15 benchmarks indicates robust few-shot generalization.",
                doi=None,
                authors=("Carol White",),
                similarity=0.74,
            ),
        ]
        self.engine = GroundedReasoningEngine(
            groq_api_key="mock_groq_key",
            gemini_api_key="mock_gemini_key",
            timeout_seconds=2.0,
        )

    def test_status_reports_model_and_keys(self) -> None:
        status = self.engine.status()
        self.assertTrue(status["groq_configured"])
        self.assertTrue(status["gemini_configured"])
        self.assertIn("llama-3.3-70b", status["groq_model"])
        self.assertEqual(status["fallback_backend"], "extractive-synthesis")

    def test_empty_papers_returns_helpful_notice(self) -> None:
        resp = self.engine.reason(ReasoningRequest(query="natural language processing", papers=()))
        self.assertEqual(resp.provider_used, "none")
        self.assertTrue(resp.is_fallback)
        self.assertIn("No relevant publications were provided", resp.answer)

    def test_extractive_fallback_when_offline(self) -> None:
        offline_engine = GroundedReasoningEngine(groq_api_key="", gemini_api_key="")
        resp = offline_engine.reason(
            ReasoningRequest(query="transformer architectures", papers=tuple(self.papers), provider="auto")
        )
        self.assertEqual(resp.provider_used, "extractive_synthesis")
        self.assertEqual(resp.model_name, "deterministic-extractive-v1")
        self.assertTrue(resp.is_fallback)
        self.assertIn("[Paper #1 · Alice Smith et al., 2024]", resp.answer)
        self.assertEqual(len(resp.citations), 2)

    @patch("requests.post")
    def test_groq_reasoning_success(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "### Grounded Answer\n\nTransformers improve translation by self-attention [Paper #1 · Smith et al., 2024]."
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        resp = self.engine.reason(
            ReasoningRequest(query="transformer models", papers=tuple(self.papers), provider="groq")
        )
        self.assertEqual(resp.provider_used, "groq")
        self.assertFalse(resp.is_fallback)
        self.assertIn("[Paper #1 · Smith et al., 2024]", resp.answer)
        self.assertIsNone(resp.error)

    @patch("requests.post")
    def test_gemini_reasoning_success(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "### Gemini Synthesis\n\nCross-lingual generalization has been established in [Paper #2 · White, 2023]."
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        resp = self.engine.reason(
            ReasoningRequest(query="multilingual evaluation", papers=tuple(self.papers), provider="gemini")
        )
        self.assertEqual(resp.provider_used, "gemini")
        self.assertFalse(resp.is_fallback)
        self.assertIn("White, 2023", resp.answer)

    @patch("requests.post")
    def test_auto_provider_fails_over_from_groq_to_gemini(self, mock_post: MagicMock) -> None:
        # First call (Groq) raises exception, second call (Gemini) succeeds
        gemini_mock = MagicMock()
        gemini_mock.status_code = 200
        gemini_mock.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Gemini answer after Groq failover [Paper #1]."}]
                    }
                }
            ]
        }

        mock_post.side_effect = [
            RuntimeError("Groq 429 Rate Limit Exceeded"),
            gemini_mock,
        ]

        resp = self.engine.reason(
            ReasoningRequest(query="deep learning", papers=tuple(self.papers), provider="auto")
        )
        self.assertEqual(resp.provider_used, "gemini")
        self.assertFalse(resp.is_fallback)
        self.assertIn("Gemini answer after Groq failover", resp.answer)

    @patch("requests.post")
    def test_both_providers_failing_falls_back_to_extractive_synthesis(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = RuntimeError("Network unreachable")

        resp = self.engine.reason(
            ReasoningRequest(query="deep learning", papers=tuple(self.papers), provider="auto")
        )
        self.assertEqual(resp.provider_used, "extractive_synthesis")
        self.assertTrue(resp.is_fallback)
        self.assertIn("Network unreachable", resp.error or "")
        self.assertIn("[Paper #1 · Alice Smith et al., 2024]", resp.answer)


if __name__ == "__main__":
    unittest.main()

