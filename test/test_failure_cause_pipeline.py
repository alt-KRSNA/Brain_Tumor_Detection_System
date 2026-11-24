
import csv
import tempfile
import unittest
from pathlib import Path

from test.failure_cause_annotation_validator import (
    validate_failure_cause_annotations,
)
from test.failure_cause_analysis import analyze_failure_causes
from test.generate_failure_cause_report import (
    generate_failure_cause_report,
)
from test.export_failure_cause_summaries import (
    export_failure_cause_summaries,
)


class TestFailureCausePipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.csv_path = self.root / "annotations.csv"
        self.report_path = self.root / "report.md"
        self.export_dir = self.root / "exports"

        self.csv_path.write_text(
            "configuration,question_id,rank,chunk_id,relevance_grade,"
            "failure_cause,review_note,review_confidence\n"
            "A,glioma,1,c1,1,related_not_answer,Related,high\n"
            "A,glioma,2,c2,0,wrong_section,Wrong,medium\n"
            "B,mri,1,c3,1,related_not_answer,Partial,high\n"
            "C,symptoms,3,c4,0,,,\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_complete_pipeline(self):
        validation = validate_failure_cause_annotations(
            self.csv_path,
            expected_total=4,
            expected_unreviewed=1,
        )

        self.assertTrue(validation["valid"])

        analysis = analyze_failure_causes(self.csv_path)

        self.assertEqual(analysis["total_records"], 4)
        self.assertEqual(analysis["reviewed_records"], 3)
        self.assertEqual(analysis["unreviewed_records"], 1)

        report = generate_failure_cause_report(
            self.csv_path,
            self.report_path,
        )

        self.assertTrue(Path(report).exists())

        exports = export_failure_cause_summaries(
            self.csv_path,
            self.export_dir,
        )

        self.assertEqual(len(exports), 4)

        for path in exports.values():
            self.assertTrue(Path(path).exists())

    def test_export_counts_match_analysis(self):
        analysis = analyze_failure_causes(self.csv_path)

        exports = export_failure_cause_summaries(
            self.csv_path,
            self.export_dir,
        )

        with Path(exports["overall"]).open(
            encoding="utf-8"
        ) as file:
            rows = list(csv.DictReader(file))

        exported_counts = {
            row["failure_cause"]: int(row["count"])
            for row in rows
        }

        self.assertEqual(
            exported_counts,
            analysis["category_counts"],
        )


if __name__ == "__main__":
    unittest.main()
