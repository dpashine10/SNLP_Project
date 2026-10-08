"""Phase 2 tests: judging template, validator and fair evaluation.

Run from the repository root::

    python -m unittest discover -s src/week5/tests -t .

Any labels set here are synthetic, exist only in memory and are never saved.
"""

import logging
import unittest
from typing import Any

import pandas as pd

from src.week5.analysis.label_agreement import agreement_report, disagreements
from src.week5.evaluation.fair_evaluation import FairEvaluationSettings, fair_evaluate
from src.week5.judging.merge import judged_levels, load_frozen_records, merge_judgments
from src.week5.judging.schema import FROZEN_COLUMNS, NEEDS_REVIEW, PHASE1_COLUMN_MAP, read_judging_csv
from src.week5.judging.template import build_template
from src.week5.judging.validation import validate_frame
from src.week5.paths import COMPLETED_JUDGMENTS_FILE, FROZEN_RUN_DIR, JUDGING_TEMPLATE, JUDGMENTS_FILE


def _judge(frame: pd.DataFrame, index: int, level: str) -> None:
    frame.loc[index, ["manual_relevance", "judge_id", "judgment_notes", "judged_at"]] = [
        level, "tester", "synthetic test label", "2026-10-08",
    ]


class JudgingTests(unittest.TestCase):
    template: pd.DataFrame
    records: list[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        logging.getLogger("src.week4").setLevel(logging.ERROR)
        cls.template = build_template(FROZEN_RUN_DIR)
        cls.records = load_frozen_records(FROZEN_RUN_DIR)

    def copy(self) -> pd.DataFrame:
        return self.template.copy()

    def errors(self, frame: pd.DataFrame) -> list[str]:
        return validate_frame(frame, self.template).errors

    def test_template_preserves_every_phase1_column(self) -> None:
        pool = pd.read_csv(FROZEN_RUN_DIR / "candidate_pool.csv", dtype=str, keep_default_na=False)
        for phase1, column in PHASE1_COLUMN_MAP.items():
            expected = pool[phase1].map({"True": "1", "False": ""}) if phase1 == "in_existing_labels" else pool[phase1]
            self.assertEqual(expected.tolist(), self.template[column].tolist(), phase1)

    def test_template_counts(self) -> None:
        self.assertEqual(len(self.template), 214)
        self.assertEqual((self.template["existing_label"] == "1").sum(), 80)
        self.assertTrue((self.template["manual_relevance"] == "").all())

    def test_saved_files_match_template_and_validate(self) -> None:
        self.assertTrue(read_judging_csv(JUDGING_TEMPLATE).equals(self.template))
        self.assertEqual(self.errors(read_judging_csv(JUDGMENTS_FILE)), [])

    @unittest.skipUnless(COMPLETED_JUDGMENTS_FILE.is_file(), "completed judging file not present")
    def test_completed_file_validates_with_frozen_columns_unchanged(self) -> None:
        completed = read_judging_csv(COMPLETED_JUDGMENTS_FILE)
        self.assertEqual(self.errors(completed), [])
        self.assertTrue(completed[list(FROZEN_COLUMNS)].equals(self.template[list(FROZEN_COLUMNS)]))

    def test_accepts_valid_levels(self) -> None:
        frame = self.copy()
        for index, level in enumerate(("0", "1", "2", "3")):
            _judge(frame, index, level)
        self.assertEqual(self.errors(frame), [])

    def test_rejects_invalid_levels(self) -> None:
        for bad in ("4", "-1", "2.5", "high", "relevant"):
            frame = self.copy()
            _judge(frame, 0, bad)
            self.assertTrue(any("manual_relevance must be" in error for error in self.errors(frame)), bad)

    def test_rejects_duplicate_pairs(self) -> None:
        frame = pd.concat([self.copy(), self.template.iloc[[0]]], ignore_index=True)
        self.assertTrue(any("Duplicate pair" in error for error in self.errors(frame)))

    def test_rejects_changed_frozen_field(self) -> None:
        frame = self.copy()
        frame.loc[3, "researcher_name"] = "Someone Else"
        self.assertTrue(any("frozen column 'researcher_name'" in error for error in self.errors(frame)))

    def test_rejects_missing_and_unknown_pairs(self) -> None:
        frame = self.copy().drop(index=5).reset_index(drop=True)
        self.assertTrue(any("Missing pair" in error for error in self.errors(frame)))
        frame = self.copy()
        frame.loc[0, "researcher_id"] = "https://openalex.org/A0000000000"
        self.assertTrue(any("not in the frozen evidence" in error for error in self.errors(frame)))

    def test_rejects_judged_row_without_required_fields(self) -> None:
        frame = self.copy()
        frame.loc[0, "manual_relevance"] = "2"
        errors = self.errors(frame)
        for column in ("judge_id", "judgment_notes", "judged_at"):
            self.assertTrue(any(f"need {column}" in error for error in errors), column)
        frame = self.copy()
        _judge(frame, 0, "2")
        frame.loc[0, "judged_at"] = "yesterday"
        self.assertTrue(any("ISO date" in error for error in self.errors(frame)))

    def test_review_flag_rules(self) -> None:
        frame = self.copy()
        frame.loc[0, ["review_flag", "judgment_notes"]] = [NEEDS_REVIEW, "titles too vague"]
        self.assertEqual(self.errors(frame), [])
        for flag in ("yes", "no"):
            _judge(frame, 1, "2")
            frame.loc[1, "review_flag"] = flag
            self.assertEqual(self.errors(frame), [], flag)
        frame.loc[2, "review_flag"] = "maybe"
        self.assertTrue(any("review_flag must be one of" in error for error in self.errors(frame)))
        frame = self.copy()
        frame.loc[0, "review_flag"] = "yes"
        self.assertTrue(any("why the row needs review" in error for error in self.errors(frame)))

    def test_exclude_flagged_treats_flagged_as_unjudged(self) -> None:
        frame = self.copy()
        _judge(frame, 0, "3")
        _judge(frame, 1, "0")
        frame.loc[1, "review_flag"] = "yes"
        key0 = (frame.loc[0, "query_id"], frame.loc[0, "researcher_id"])
        self.assertEqual(len(judged_levels(frame)), 2)
        self.assertEqual(judged_levels(frame, exclude_flagged=True), {key0: 3})

    def test_label_agreement_counts(self) -> None:
        frame = self.copy()
        for index in frame.index:
            _judge(frame, int(index), "3" if frame.loc[index, "existing_label"] == "1" else "0")
        report = agreement_report(frame)
        self.assertEqual(report["overall"]["differ"], 0)
        frame.loc[frame["existing_label"] == "", "manual_relevance"] = "2"
        report = agreement_report(frame)
        self.assertEqual(report["explicit_old_labels"]["differ"], 0)
        self.assertEqual(report["implied_old_negatives"]["differ"], 134)
        self.assertEqual(len(disagreements(frame)), 134)

    def test_rejects_missing_columns(self) -> None:
        frame = self.copy().drop(columns=["judge_id"])
        self.assertTrue(any("Missing required columns" in error for error in self.errors(frame)))

    def test_blank_is_unjudged_not_irrelevant(self) -> None:
        self.assertEqual(judged_levels(self.template), {})
        merged = merge_judgments(self.records, self.template)
        self.assertTrue(all(entry["manual_relevance"] is None for record in merged for entry in record["ranking"]))
        result = fair_evaluate(self.records, {})
        self.assertFalse(result.ready)
        self.assertEqual((result.judged_pairs, result.total_pairs), (0, 214))
        self.assertEqual(result.summary, [])


class FairEvaluationTests(unittest.TestCase):
    """Synthetic in-memory labels check the evaluation rules; they are not results."""

    records: list[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_frozen_records(FROZEN_RUN_DIR)

    def pool(self, query_id: str) -> list[str]:
        authors: dict[str, None] = {}
        for record in self.records:
            if record["query_id"] == query_id:
                authors.update((entry["author_id"], None) for entry in record["ranking"])
        return list(authors)

    def test_only_fully_judged_queries_are_evaluated(self) -> None:
        authors = self.pool("q05")
        labels = {("q05", author): (3 if position % 2 == 0 else 0) for position, author in enumerate(authors)}
        result = fair_evaluate(self.records, labels)
        self.assertEqual(result.evaluated_queries, ["q05"])
        self.assertEqual(len(result.excluded_queries), 7)
        self.assertEqual(len(result.summary), 6)
        self.assertTrue(all(row["unjudged_removed"] == 0 for row in result.per_query))

    def test_unjudged_researchers_are_removed_not_counted_wrong(self) -> None:
        tfidf = next(r for r in self.records if r["system"] == "Week 3 (tfidf)" and r["query_id"] == "q01")
        ranked = [entry["author_id"] for entry in tfidf["ranking"]]
        labels = {("q01", author): 3 for author in ranked[5:]}
        labels.update({("q01", author): 0 for author in self.pool("q01") if author not in ranked})
        settings = FairEvaluationSettings(min_judged_fraction=0.5)
        row = next(r for r in fair_evaluate(self.records, labels, settings).per_query if r["system"] == "Week 3 (tfidf)")
        self.assertEqual(row["unjudged_removed"], 5)
        self.assertEqual(row["mrr"], 1.0)

    def test_threshold_controls_binary_relevance(self) -> None:
        labels = {("q05", author): 1 for author in self.pool("q05")}
        self.assertIn("q05", fair_evaluate(self.records, labels).excluded_queries)
        loose = FairEvaluationSettings(relevant_threshold=1)
        self.assertEqual(fair_evaluate(self.records, labels, loose).evaluated_queries, ["q05"])

    def test_invalid_settings_are_rejected(self) -> None:
        for kwargs in ({"min_judged_fraction": 0.0}, {"min_judged_fraction": 1.5}, {"relevant_threshold": 4}):
            with self.assertRaises(ValueError):
                FairEvaluationSettings(**kwargs)


if __name__ == "__main__":
    unittest.main()
