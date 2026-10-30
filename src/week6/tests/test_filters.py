"""Unit tests for document-level non-research artifact filtering."""

import unittest

from src.week6.filters import is_non_research_artifact


class TestNonResearchArtifactFilter(unittest.TestCase):
    def test_excludes_documented_week5_failure_modes(self) -> None:
        excluded_titles = (
            None,
            "",
            "   ",
            "Unknown Title",
            "Editorial",
            "Guest Editorial",
            "Editorial: Advances in Information Retrieval",
            "Retraction Note to: Deep Learning for Biomedical NLP",
            "Retraction Note: Transformer Pretraining",
            "Retracted: Graph Neural Networks",
            "Erratum: Quantum Error Mitigation",
            "Erratum to: Sequence Labeling",
            "Corrigendum to Statistical Parsing",
            "Publisher Correction: Neural Machine Translation",
            "Author Correction: Speech Recognition",
            "Code for the ML models",
            "Supplementary code for benchmark experiments",
            "Replication code for causal inference study",
        )
        for title in excluded_titles:
            self.assertTrue(is_non_research_artifact(title), f"Expected exclusion for {title!r}")

    def test_preserves_legitimate_scholarly_titles(self) -> None:
        valid_titles = (
            "Code-Switching and Code-Mixing in Multilingual Transformers",
            "Code Generation with Large Language Models",
            "Editorial Stance Detection in News Articles",
            "Error Correction Codes for Fault-Tolerant Quantum Computing",
            "Retractable Roof Structures under Wind Loads",
            "Attention Is All You Need",
        )
        for title in valid_titles:
            self.assertFalse(is_non_research_artifact(title), f"Expected preservation for {title!r}")


if __name__ == "__main__":
    unittest.main()

