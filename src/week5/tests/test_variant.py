"""Tests for the improved Week 4 variant and scoring a system against the frozen pool.

Run from the repository root::

    python -m unittest discover -s src/week5/tests -t .

Any labels set here are synthetic, exist only in memory and are never saved.
"""

import copy
import dataclasses
import unittest
from typing import Any

from src.week4.config import CONFIG
from src.week4.ranking_strategy import TopKAverageAuthorDiscountStrategy, TopKAverageStrategy, create_strategy
from src.week4.schemas import Author, Paper, PaperHit
from src.week5.analysis.robustness import VARIANT_QUERIES, jaccard, variant_rows, variant_summary
from src.week5.evaluation.fair_evaluation import fair_evaluate
from src.week5.evaluation.variant_evaluation import IMPROVED_STRATEGY, unjudged_pairs
from src.week5.judging.merge import load_frozen_records
from src.week5.paths import FROZEN_RUN_DIR


def _hit(score: float, author_count: int) -> PaperHit:
    authors = tuple(Author(author_id=f"a{index}", name=f"Author {index}") for index in range(author_count))
    return PaperHit(paper=Paper(paper_id=f"p{score}-{author_count}", title="t", text="t", authors=authors), score=score)


class AuthorDiscountStrategyTests(unittest.TestCase):
    def test_single_author_papers_match_top_k_average(self) -> None:
        hits = [_hit(0.9, 1), _hit(0.7, 1), _hit(0.5, 1), _hit(0.1, 1)]
        self.assertAlmostEqual(TopKAverageAuthorDiscountStrategy(k=3).score(hits), TopKAverageStrategy(k=3).score(hits))

    def test_score_is_divided_by_sqrt_of_author_count(self) -> None:
        self.assertAlmostEqual(TopKAverageAuthorDiscountStrategy(k=3).score([_hit(0.8, 4)]), 0.4)
        self.assertAlmostEqual(TopKAverageAuthorDiscountStrategy(k=1).score([_hit(0.9, 9)]), 0.3)

    def test_discount_is_applied_before_choosing_the_top_k(self) -> None:
        # The 16-author paper has the highest raw score but the lowest discounted one.
        hits = [_hit(1.0, 16), _hit(0.6, 1), _hit(0.5, 1)]
        self.assertAlmostEqual(TopKAverageAuthorDiscountStrategy(k=2).score(hits), 0.55)

    def test_paper_without_authors_is_not_discounted(self) -> None:
        self.assertAlmostEqual(TopKAverageAuthorDiscountStrategy(k=1).score([_hit(0.6, 0)]), 0.6)

    def test_rejects_empty_hits_and_invalid_k(self) -> None:
        with self.assertRaises(ValueError):
            TopKAverageAuthorDiscountStrategy(k=3).score([])
        with self.assertRaises(ValueError):
            TopKAverageAuthorDiscountStrategy(k=0)

    def test_registered_with_config_k(self) -> None:
        config = dataclasses.replace(CONFIG, ranking_strategy=IMPROVED_STRATEGY, strategy_top_k=2)
        strategy = create_strategy(config)
        self.assertIsInstance(strategy, TopKAverageAuthorDiscountStrategy)
        self.assertAlmostEqual(strategy.score([_hit(0.8, 4), _hit(0.4, 1), _hit(0.2, 1)]), 0.4)


class PoolRecordsTests(unittest.TestCase):
    """A later system is scored on the frozen pool; its extra researchers are not judged."""

    records: list[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_frozen_records(FROZEN_RUN_DIR)

    def new_system(self, query_id: str) -> dict[str, Any]:
        """A copy of a tfidf ranking with two researchers swapped for unknown ones."""
        base = next(r for r in self.records if r["system"] == "Week 3 (tfidf)" and r["query_id"] == query_id)
        record = copy.deepcopy(base)
        record["system"] = "New system"
        for position in (0, 1):
            record["ranking"][position]["author_id"] = f"unknown-{position}"
        return record

    def labels(self, query_id: str) -> dict[tuple[str, str], int]:
        pool = {e["author_id"] for r in self.records if r["query_id"] == query_id for e in r["ranking"]}
        return {(query_id, author): 3 for author in pool}

    def test_pool_records_keep_eligibility_and_remove_unjudged(self) -> None:
        extra = self.new_system("q05")
        labels = self.labels("q05")
        # Without pool_records the unknown researchers enlarge the pool, so q05 is no longer fully judged.
        self.assertNotIn("q05", fair_evaluate([*self.records, extra], labels).evaluated_queries)

        result = fair_evaluate([*self.records, extra], labels, pool_records=self.records)
        self.assertEqual(result.evaluated_queries, ["q05"])
        self.assertEqual(result.total_pairs, fair_evaluate(self.records, labels).total_pairs)
        row = next(r for r in result.per_query if r["system"] == "New system")
        self.assertEqual((row["unjudged_removed"], row["judged_results"], row["relevant_results"]), (2, 8, 8))
        summary = next(s for s in result.summary if s["system"] == "New system")
        self.assertEqual((summary["judged_results"], summary["precision_among_judged"]), (8, 1.0))

    def test_unjudged_pairs_lists_each_missing_pair_once(self) -> None:
        extra = self.new_system("q05")
        missing = unjudged_pairs([extra, extra], set(self.labels("q05")))
        self.assertEqual([row["researcher_id"] for row in missing], ["unknown-0", "unknown-1"])
        self.assertTrue(all(row["manual_relevance"] == "" for row in missing))


class RobustnessTests(unittest.TestCase):
    """Synthetic rankings check the robustness measures; they are not results."""

    def test_jaccard(self) -> None:
        self.assertEqual(jaccard(["a", "b"], ["b", "c"]), 1 / 3)
        self.assertEqual(jaccard([], []), 1.0)
        self.assertEqual(jaccard([str(i) for i in range(12)], [str(i) for i in range(10, 20)]), 0.0)

    def test_rephrasings_use_base_judgments_and_skip_unjudged(self) -> None:
        base_ids = sorted({query.base_query_id for query in VARIANT_QUERIES if query.base_query_id})
        rankings = {("S", base_id): ["a", "b"] for base_id in base_ids}
        rankings.update({("S", query.query_id): ["a", "x"] for query in VARIANT_QUERIES})
        judgments = {(base_id, author): level for base_id in base_ids for author, level in (("a", 3), ("b", 0))}
        rows = variant_rows(rankings, {base_id: base_id for base_id in base_ids}, judgments)
        rephrased = next(row for row in rows if row["kind"] == "short_keyword")
        self.assertEqual((rephrased["shared_with_base"], rephrased["judged_results"]), (1, 1))
        self.assertEqual((rephrased["precision_among_judged"], rephrased["same_top1"]), (1.0, True))
        original = next(s for s in variant_summary(rows) if s["kind"] == "original")
        self.assertEqual((original["mean_jaccard_with_base"], original["precision_among_judged"]), (1.0, 0.5))


if __name__ == "__main__":
    unittest.main()
