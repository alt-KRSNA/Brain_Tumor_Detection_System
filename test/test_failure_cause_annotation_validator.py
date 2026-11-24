# test/test_failure_cause_annotation_validator.py

import csv
import tempfile
import unittest
from pathlib import Path

from test.failure_cause_annotation_validator import (
    validate_failure_cause_annotations,
)


class TestFailureCauseAnnotationValidator(unittest.TestCase):

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
                "chunk_id": "chunk_32",
                "relevance_grade": "1",
                "failure_cause": "related_not_answer",
                "review_note": "Related subtype information.",
                "review_confidence": "high",
            },
            {
                "configuration": "A",
                "question_id": "glioma",
                "rank": "3",
                "chunk_id": "chunk_74",
                "relevance_grade": "0",
                "failure_cause": "",
                "review_note": "",
                "review_confidence": "",
            },
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_csv(self, rows):
        with self.csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=self.fieldnames
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_valid_annotations_pass(self):
        self.write_csv(self.rows)

        result = validate_failure_cause_annotations(
            self.csv_path,
            expected_total=2,
            expected_unreviewed=1,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["total_records"], 2)
        self.assertEqual(result["reviewed_records"], 1)
        self.assertEqual(result["unreviewed_records"], 1)

    def test_invalid_failure_cause_fails(self):
        rows = [dict(row) for row in self.rows]
        rows[0]["failure_cause"] = "invalid_category"

        self.write_csv(rows)

        result = validate_failure_cause_annotations(
            self.csv_path,
            expected_total=2,
            expected_unreviewed=1,
        )

        self.assertFalse(result["valid"])

    def test_duplicate_identifiers_fail(self):
        rows = [dict(row) for row in self.rows]
        rows[1]["rank"] = rows[0]["rank"]
        rows[1]["chunk_id"] = rows[0]["chunk_id"]
        rows[1]["question_id"] = rows[0]["question_id"]

        self.write_csv(rows)

        result = validate_failure_cause_annotations(
            self.csv_path,
            expected_total=2,
            expected_unreviewed=0,
        )

        self.assertFalse(result["valid"])
        self.assertTrue(
            any("Duplicate" in error for error in result["errors"])
        )

    def test_partial_annotation_fails(self):
        rows = [dict(row) for row in self.rows]
        rows[1]["failure_cause"] = "wrong_section"

        self.write_csv(rows)

        result = validate_failure_cause_annotations(
            self.csv_path,
            expected_total=2,
            expected_unreviewed=1,
        )

        self.assertFalse(result["valid"])

    def test_invalid_relevance_grade_fails(self):
        rows = [dict(row) for row in self.rows]
        rows[0]["relevance_grade"] = "2"

        self.write_csv(rows)

        result = validate_failure_cause_annotations(
            self.csv_path,
            expected_total=2,
            expected_unreviewed=1,
        )

        self.assertFalse(result["valid"])


if __name__ == "__main__":
    unittest.main()