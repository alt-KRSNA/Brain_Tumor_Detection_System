
import tempfile
import unittest
from pathlib import Path

from test.generate_failure_investigation_report import (
    build_markdown,
    write_failure_csv,
)


class TestFailureInvestigationReport(unittest.TestCase):

    def setUp(self):
        self.report = {
            "overall": {
                "total": 2,
                "grade_0": 1,
                "grade_1": 1,
                "grade_2": 0,
                "failure_count": 2,
                "failure_rate": 1.0,
                "average_grade": 0.5,
                "average_distance": 0.4,
            },
            "failure_count": 2,
            "failure_rate": 1.0,
            "by_configuration": {},
            "by_question": {},
            "by_source": {},
            "by_rank": {},
            "failure_details": [
                {
                    "annotation_key": "A|glioma|1",
                    "failure_type": "irrelevant",
                    "configuration": "A",
                    "question_id": "glioma",
                    "question": "What is glioma?",
                    "rank": 1,
                    "relevance_grade": 0,
                    "chunk_id": "chunk_1",
                    "source": "glioma.pdf",
                    "page": 5,
                    "distance": 0.4,
                    "text": "Example passage.",
                }
            ],
            "repeated_chunks": [],
            "notes": ["Test limitation note."],
        }

    def test_markdown_contains_failure_evidence(self):
        markdown = build_markdown(self.report)

        self.assertIn("# Retrieval Failure Investigation", markdown)
        self.assertIn("A|glioma|1", markdown)
        self.assertIn("Example passage.", markdown)
        self.assertIn("Grade 0", markdown)

    def test_csv_is_generated(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure_evidence.csv"

            write_failure_csv(
                self.report["failure_details"],
                output,
            )

            self.assertTrue(output.exists())

            content = output.read_text(encoding="utf-8-sig")

            self.assertIn("annotation_key", content)
            self.assertIn("A|glioma|1", content)
            self.assertIn("Example passage.", content)


if __name__ == "__main__":
    unittest.main()
