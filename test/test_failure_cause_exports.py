
import csv
import tempfile
import unittest
from pathlib import Path

from test.export_failure_cause_summaries import export_failure_cause_summaries


class TestFailureCauseExports(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.input_csv = self.root / "annotations.csv"
        self.output_dir = self.root / "exports"

        self.input_csv.write_text(
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

    def test_all_four_exports_are_created(self):
        paths = export_failure_cause_summaries(
            self.input_csv,
            self.output_dir,
        )

        self.assertEqual(len(paths), 4)

        for path in paths.values():
            self.assertTrue(Path(path).exists())

    def test_overall_category_counts(self):
        export_failure_cause_summaries(
            self.input_csv,
            self.output_dir,
        )

        path = self.output_dir / "failure_cause_summary.csv"

        with path.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        counts = {
            row["failure_cause"]: int(row["count"])
            for row in rows
        }

        self.assertEqual(counts["related_not_answer"], 2)
        self.assertEqual(counts["wrong_section"], 1)

    def test_configuration_export(self):
        export_failure_cause_summaries(
            self.input_csv,
            self.output_dir,
        )

        path = self.output_dir / "failure_cause_by_configuration.csv"

        with path.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        self.assertTrue(any(row["configuration"] == "A" for row in rows))

    def test_question_export(self):
        export_failure_cause_summaries(
            self.input_csv,
            self.output_dir,
        )

        path = self.output_dir / "failure_cause_by_question.csv"

        with path.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        self.assertTrue(any(row["question_id"] == "glioma" for row in rows))

    def test_confidence_export(self):
        export_failure_cause_summaries(
            self.input_csv,
            self.output_dir,
        )

        path = self.output_dir / "failure_cause_by_confidence.csv"

        with path.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        counts = {
            row["confidence"]: int(row["count"])
            for row in rows
        }

        self.assertEqual(counts["high"], 2)
        self.assertEqual(counts["medium"], 1)


if __name__ == "__main__":
    unittest.main()
