import csv
import tempfile
import unittest
from pathlib import Path

from test.failure_cause_template import generate_failure_cause_template


class TestFailureCauseTemplate(unittest.TestCase):

    def test_generates_template_for_grade_zero_and_one(self):
        records = [
            {
                "configuration": "A",
                "question_id": "glioma",
                "rank": 1,
                "chunk_id": "chunk_1",
                "source": "glioma.pdf",
                "page": 5,
                "relevance_grade": 1,
            },
            {
                "configuration": "A",
                "question_id": "glioma",
                "rank": 2,
                "chunk_id": "chunk_2",
                "source": "glioma.pdf",
                "page": 6,
                "relevance_grade": 2,
            },
            {
                "configuration": "B",
                "question_id": "mri",
                "rank": 3,
                "chunk_id": "chunk_3",
                "source": "brain_mri_classification.pdf",
                "page": 4,
                "relevance_grade": 0,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "annotations.csv"

            count = generate_failure_cause_template(
                records,
                output_path
            )

            with output_path.open(
                "r",
                encoding="utf-8",
                newline=""
            ) as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(count, 2)
        self.assertEqual(len(rows), 2)

        grades = [int(row["relevance_grade"]) for row in rows]

        self.assertEqual(grades, [1, 0])

    def test_template_contains_annotation_columns(self):
        records = [
            {
                "configuration": "A",
                "question_id": "glioma",
                "rank": 1,
                "chunk_id": "chunk_1",
                "source": "glioma.pdf",
                "page": 5,
                "relevance_grade": 1,
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "annotations.csv"

            generate_failure_cause_template(
                records,
                output_path
            )

            with output_path.open(
                "r",
                encoding="utf-8",
                newline=""
            ) as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                columns = reader.fieldnames

        expected_columns = [
            "configuration",
            "question_id",
            "rank",
            "chunk_id",
            "source",
            "page",
            "relevance_grade",
            "failure_cause",
            "review_note",
            "review_confidence",
        ]

        self.assertEqual(columns, expected_columns)
        self.assertEqual(rows[0]["failure_cause"], "")
        self.assertEqual(rows[0]["review_note"], "")
        self.assertEqual(rows[0]["review_confidence"], "")


if __name__ == "__main__":
    unittest.main()