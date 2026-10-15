"""Unit tests for the confidence calibration engine."""

import unittest

from src.week6.confidence import ConfidenceCalibrator
from src.week6.schemas import ConfidenceLevel


class TestConfidenceCalibrator(unittest.TestCase):
    def setUp(self) -> None:
        self.calibrator = ConfidenceCalibrator(strategy="top_k_average_author_discount")

    def test_high_score_produces_high_confidence(self) -> None:
        conf = self.calibrator.calibrate(
            raw_score=0.60,
            rank=1,
            matched_papers=3,
            next_score=0.45,
        )
        self.assertEqual(conf.level, ConfidenceLevel.HIGH.value)
        self.assertGreaterEqual(conf.score, 0.75)
        self.assertAlmostEqual(conf.margin, 0.15, places=3)
        self.assertIn("High confidence", conf.explanation)

    def test_low_score_produces_low_confidence(self) -> None:
        conf = self.calibrator.calibrate(
            raw_score=0.15,
            rank=5,
            matched_papers=1,
            next_score=0.14,
        )
        self.assertEqual(conf.level, ConfidenceLevel.LOW.value)
        self.assertLess(conf.score, 0.50)
        self.assertIn("Low confidence", conf.explanation)

    def test_medium_score_produces_medium_confidence(self) -> None:
        conf = self.calibrator.calibrate(
            raw_score=0.33,
            rank=2,
            matched_papers=1,
            next_score=0.32,
        )
        self.assertEqual(conf.level, ConfidenceLevel.MEDIUM.value)
        self.assertGreaterEqual(conf.score, 0.50)
        self.assertLess(conf.score, 0.75)

    def test_score_clamping(self) -> None:
        # Extreme high score
        extreme_high = self.calibrator.calibrate(
            raw_score=1.5,
            rank=1,
            matched_papers=10,
            next_score=0.0,
        )
        self.assertLessEqual(extreme_high.score, 0.98)

        # Extreme low score
        extreme_low = self.calibrator.calibrate(
            raw_score=-1.0,
            rank=10,
            matched_papers=0,
            next_score=None,
        )
        self.assertGreaterEqual(extreme_low.score, 0.05)

    def test_evidence_bonus(self) -> None:
        conf_single = self.calibrator.calibrate(
            raw_score=0.35, rank=2, matched_papers=1, next_score=0.30
        )
        conf_multi = self.calibrator.calibrate(
            raw_score=0.35, rank=2, matched_papers=4, next_score=0.30
        )
        self.assertGreater(conf_multi.score, conf_single.score)


if __name__ == "__main__":
    unittest.main()
