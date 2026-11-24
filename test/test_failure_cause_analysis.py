import csv
import tempfile
import unittest
from pathlib import Path

from test.failure_cause_analysis import analyze_failure_causes


class TestFailureCauseAnalysis(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "annotations.csv"

        self.fieldnames = [
            "configuration",
            "question_id",
            "rank",
            "chunk_id",
            "relevance_grade",
            "failure_cause",
            "review_note",
            "review_confidence",
        ]

        self.rows = [
            {
                "configuration": "A",
                "question_id": "glioma",
                "rank": "1",
                "chunk_id": "chunk_1",
                "relevance_grade": "1",
                "failure_cause": "related_not_answer",
                "review_note": "Related information.",
                "review_confidence": "high",
            },
            {
                "configuration": "A",
                "question_id": "glioma",
                "rank": "2",
                "chunk_id": "chunk_2",
                "relevance_grade": "0",
                "failure_cause": "wrong_section",
                "review_note": "Wrong section.",
                "review_confidence": "medium",
            },
            {
                "configuration": "B",
                "question_id": "mri",
                "rank": "1",
                "chunk_id": "chunk_3",
                "relevance_grade": "1",
                "failure_cause": "related_not_answer",
                "review_note": "Partially relevant.",
                "review_confidence": "high",
            },
            {
                "configuration": "C",
                "question_id": "symptoms",
                "rank": "3",
                "chunk_id": "chunk_4",
                "relevance_grade": "0",
                "failure_cause": "",
                "review_note": "",
                "review_confidence": "",
            },
        ]

        with self.csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=self.fieldnames
            )
            writer.writeheader()
            writer.writerows(self.rows)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_total_and_reviewed_counts(self):
        result = analyze_failure_causes(self.csv_path)

        self.assertEqual(result["total_records"], 4)
        self.assertEqual(result["reviewed_records"], 3)
        self.assertEqual(result["unreviewed_records"], 1)

    def test_category_distribution(self):
        result = analyze_failure_causes(self.csv_path)

        self.assertEqual(
            result["category_counts"]["related_not_answer"], 2
        )
        self.assertEqual(
            result["category_counts"]["wrong_section"], 1
        )

    def test_configuration_distribution(self):
        result = analyze_failure_causes(self.csv_path)

        self.assertEqual(
            result["configuration_counts"]["A"]["related_not_answer"],
            1,
        )
        self.assertEqual(
            result["configuration_counts"]["A"]["wrong_section"],
            1,
        )
        self.assertEqual(
            result["configuration_counts"]["B"]["related_not_answer"],
            1,
        )

    def test_question_distribution(self):
        result = analyze_failure_causes(self.csv_path)

        self.assertEqual(
            result["question_counts"]["glioma"]["related_not_answer"],
            1,
        )
        self.assertEqual(
            result["question_counts"]["mri"]["related_not_answer"],
            1,
        )

    def test_confidence_distribution(self):
        result = analyze_failure_causes(self.csv_path)

        self.assertEqual(result["confidence_counts"]["high"], 2)
        self.assertEqual(result["confidence_counts"]["medium"], 1)


if __name__ == "__main__":
    unittest.main()