import unittest

from test.evaluate_passage_relevance import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    average_relevance_at_k,
    evaluate_question,
)


class TestPassageMetrics(unittest.TestCase):

    def setUp(self):

        self.grades = [2, 0, 1, 0, 2]

    def test_precision_at_3(self):

        self.assertAlmostEqual(
            precision_at_k(self.grades, 3),
            2 / 3
        )

    def test_precision_at_5(self):

        self.assertAlmostEqual(
            precision_at_k(self.grades, 5),
            3 / 5
        )

    def test_recall_at_3(self):

        self.assertAlmostEqual(
            recall_at_k(self.grades, 3),
            2 / 3
        )

    def test_recall_at_5(self):

        self.assertAlmostEqual(
            recall_at_k(self.grades, 5),
            1.0
        )

    def test_mrr_first_relevant_at_rank_1(self):

        self.assertAlmostEqual(
            mean_reciprocal_rank(self.grades),
            1.0
        )

    def test_mrr_first_relevant_at_rank_2(self):

        grades = [0, 2, 1, 0, 0]

        self.assertAlmostEqual(
            mean_reciprocal_rank(grades),
            0.5
        )

    def test_mrr_no_relevant_passages(self):

        grades = [0, 0, 0, 0, 0]

        self.assertAlmostEqual(
            mean_reciprocal_rank(grades),
            0.0
        )

    def test_ndcg_perfect_ranking(self):

        grades = [2, 2, 1, 0, 0]

        self.assertAlmostEqual(
            ndcg_at_k(grades, 5),
            1.0
        )

    def test_ndcg_worse_ranking(self):

        actual = [0, 1, 0, 2, 0]
        ideal = [2, 1, 0, 0, 0]

        self.assertLess(
            ndcg_at_k(actual, 3),
            ndcg_at_k(ideal, 3)
        )

    def test_ndcg_all_irrelevant(self):

        grades = [0, 0, 0, 0, 0]

        self.assertAlmostEqual(
            ndcg_at_k(grades, 5),
            0.0
        )

    def test_average_relevance_at_3(self):

        self.assertAlmostEqual(
            average_relevance_at_k(self.grades, 3),
            1.0
        )

    def test_average_relevance_at_5(self):

        self.assertAlmostEqual(
            average_relevance_at_k(self.grades, 5),
            1.0
        )

    def test_question_evaluation_returns_all_metrics(self):

        metrics = evaluate_question(
            self.grades
        )

        expected_metrics = [
            "MRR",
            "Precision@3",
            "Recall@3",
            "nDCG@3",
            "AvgRelevance@3",
            "Precision@5",
            "Recall@5",
            "nDCG@5",
            "AvgRelevance@5",
        ]

        for metric in expected_metrics:

            self.assertIn(
                metric,
                metrics
            )

    def test_precision_zero_when_no_passages(self):

        self.assertEqual(
            precision_at_k([], 3),
            0.0
        )


if __name__ == "__main__":
    unittest.main()