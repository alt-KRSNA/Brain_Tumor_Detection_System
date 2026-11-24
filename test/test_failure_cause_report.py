
import tempfile
import unittest
from pathlib import Path

from test.generate_failure_cause_report import generate_failure_cause_report


class TestFailureCauseReport(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.csv_path = self.root / "annotations.csv"
        self.report_path = self.root / "failure_cause_report.md"

        self.csv_path.write_text(
            "configuration,question_id,rank,chunk_id,relevance_grade,"
            "failure_cause,review_note,review_confidence\n"
            "A,glioma,1,chunk_1,1,related_not_answer,"
            "Related information,high\n"
            "A,glioma,2,chunk_2,0,wrong_section,"
            "Wrong section,medium\n"
            "B,mri,1,chunk_3,1,related_not_answer,"
            "Partial answer,high\n"
            "C,symptoms,3,chunk_4,0,,,\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_report_is_generated(self):
        generate_failure_cause_report(
            self.csv_path,
            self.report_path,
        )

        self.assertTrue(self.report_path.exists())

    def test_report_contains_overall_statistics(self):
        generate_failure_cause_report(
            self.csv_path,
            self.report_path,
        )

        report = self.report_path.read_text(encoding="utf-8")

        self.assertIn("Total records", report)
        self.assertIn("4", report)
        self.assertIn("Reviewed records", report)
        self.assertIn("3", report)
        self.assertIn("Unreviewed records", report)

    def test_report_contains_failure_categories(self):
        generate_failure_cause_report(
            self.csv_path,
            self.report_path,
        )

        report = self.report_path.read_text(encoding="utf-8")

        self.assertIn("related_not_answer", report)
        self.assertIn("wrong_section", report)

    def test_report_contains_configuration_and_question_analysis(self):
        generate_failure_cause_report(
            self.csv_path,
            self.report_path,
        )

        report = self.report_path.read_text(encoding="utf-8")

        self.assertIn("Configuration-wise", report)
        self.assertIn("Question-wise", report)
        self.assertIn("glioma", report)
        self.assertIn("mri", report)

    def test_report_contains_confidence_distribution(self):
        generate_failure_cause_report(
            self.csv_path,
            self.report_path,
        )

        report = self.report_path.read_text(encoding="utf-8")

        self.assertIn("high", report)
        self.assertIn("medium", report)
        self.assertIn("Confidence", report)


if __name__ == "__main__":
    unittest.main()
