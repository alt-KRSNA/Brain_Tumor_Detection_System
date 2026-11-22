
"""Tests for retrieval failure report generation."""

import csv
import tempfile
import unittest
from pathlib import Path

from test.failure_analysis import analyze_annotations
from test.generate_failure_analysis_report import (
    export_failure_records_csv,
    export_summary_csv,
    generate_markdown_report,
)


def sample_annotations():
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
            "relevance_grade": 0,
        },
        "B|glioma|1": {
            "configuration": "B",
            "question_id": "glioma",
            "rank": 1,
            "relevance_grade": 1,
        },
    }


class TestFailureAnalysisReport(unittest.TestCase):

    def setUp(self):
        self.results = analyze_annotations(
            sample_annotations()
        )

        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_markdown_report_generated(self):
        output = generate_markdown_report(
            self.results,
            self.output_dir / "report.md",
        )

        self.assertTrue(output.exists())

        content = output.read_text(encoding="utf-8")

        self.assertIn(
            "Retrieval Failure Analysis Report",
            content,
        )
        self.assertIn("Configuration Comparison", content)
        self.assertIn("Failure Records", content)
        self.assertIn("provisional", content.lower())

    def test_failure_csv_generated(self):
        output = export_failure_records_csv(
            self.results,
            self.output_dir / "failures.csv",
        )

        with output.open(
            newline="", encoding="utf-8"
        ) as file:
            rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 2)

    def test_failure_csv_contains_expected_columns(self):
        output = export_failure_records_csv(
            self.results,
            self.output_dir / "failures.csv",
        )

        expected = {
            "annotation_key",
            "configuration",
            "question_id",
            "rank",
            "relevance_grade",
            "grade_label",
        }

        with output.open(
            newline="", encoding="utf-8"
        ) as file:
            reader = csv.DictReader(file)

            self.assertEqual(
                set(reader.fieldnames),
                expected,
            )

    def test_summary_csv_generated(self):
        output = export_summary_csv(
            self.results,
            self.output_dir / "summary.csv",
        )

        with output.open(
            newline="", encoding="utf-8"
        ) as file:
            rows = list(csv.DictReader(file))

        self.assertGreater(len(rows), 0)

        types = {
            row["group_type"] for row in rows
        }

        self.assertEqual(
            types,
            {"configuration", "query", "rank"},
        )


if __name__ == "__main__":
    unittest.main()
