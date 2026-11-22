
"""Tests for retrieval failure analysis."""

import unittest

from test.failure_analysis import analyze_annotations


def build_annotations():
    return {
        "A|glioma|1": {
            "configuration": "A",
            "question_id": "glioma",
            "rank": 1,
            "relevance_grade": 2,
        },
        "A|glioma|2": {
            "configuration": "A",
            "question_id": "glioma",
            "rank": 2,
            "relevance_grade": 1,
        },
        "A|glioma|3": {
            "configuration": "A",
            "question_id": "glioma",
            "rank": 3,
            "relevance_grade": 0,
        },
        "B|glioma|1": {
            "configuration": "B",
            "question_id": "glioma",
            "rank": 1,
            "relevance_grade": 2,
        },
        "B|glioma|2": {
            "configuration": "B",
            "question_id": "glioma",
            "rank": 2,
            "relevance_grade": 2,
        },
        "B|glioma|3": {
            "configuration": "B",
            "question_id": "glioma",
            "rank": 3,
            "relevance_grade": 2,
        },
    }


class TestFailureAnalysis(unittest.TestCase):

    def setUp(self):
        self.annotations = build_annotations()
        self.results = analyze_annotations(self.annotations)

    def test_total_passage_count(self):
        self.assertEqual(
            self.results["overall"]["total_passages"],
            6,
        )

    def test_grade_counts(self):
        overall = self.results["overall"]

        self.assertEqual(overall["grade_0_count"], 1)
        self.assertEqual(overall["grade_1_count"], 1)
        self.assertEqual(overall["grade_2_count"], 4)

    def test_failure_count(self):
        self.assertEqual(
            self.results["overall"]["failure_count"],
            2,
        )

    def test_failure_rate(self):
        self.assertAlmostEqual(
            self.results["overall"]["failure_rate"],
            2 / 6,
        )

    def test_configuration_failure_rates(self):
        configs = self.results["configuration_analysis"]

        self.assertAlmostEqual(
            configs["A"]["failure_rate"],
            2 / 3,
        )

        self.assertEqual(
            configs["B"]["failure_rate"],
            0.0,
        )

    def test_worst_configuration(self):
        self.assertEqual(
            self.results["worst_configuration"],
            "A",
        )

    def test_best_configuration(self):
        self.assertEqual(
            self.results["best_configuration"],
            "B",
        )

    def test_failure_records_only_grade_zero_and_one(self):
        failures = self.results["failure_records"]

        self.assertEqual(len(failures), 2)

        grades = {
            item["relevance_grade"] for item in failures
        }

        self.assertEqual(grades, {0, 1})

    def test_rank_analysis(self):
        rank_analysis = self.results["rank_analysis"]

        self.assertEqual(rank_analysis[1]["grade_2"], 2)
        self.assertEqual(rank_analysis[2]["grade_1"], 1)
        self.assertEqual(rank_analysis[3]["grade_0"], 1)

    def test_query_analysis(self):
        query = self.results["query_analysis"]["glioma"]

        self.assertEqual(query["total"], 6)
        self.assertEqual(query["failure_count"], 2)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            analyze_annotations({})


if __name__ == "__main__":
    unittest.main()
