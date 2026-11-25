import csv
import tempfile
import unittest

from pathlib import Path

from test.create_baseline_relevance_template import (
    create_relevance_template,
)


class TestBaselineRelevanceTemplate(unittest.TestCase):

    def setUp(self):
        self.sample_rows = [
            {
                "configuration": "A - Baseline",
                "question_id": "Q01",
                "question": "What is glioma?",
                "expected_sources": "glioma.pdf",
                "rank": "1",
                "chunk_id": "chunk_1",
                "source": "glioma.pdf",
                "page": "4",
                "distance": "0.3490",
                "text": "Glioma passage content.",
            },
            {
                "configuration": "B - Medium",
                "question_id": "Q01",
                "question": "What is glioma?",
                "expected_sources": "glioma.pdf",
                "rank": "1",
                "chunk_id": "chunk_2",
                "source": "glioma.pdf",
                "page": "5",
                "distance": "0.3179",
                "text": "Another glioma passage.",
            },
        ]

        self.fields = list(
            self.sample_rows[0].keys()
        )

    def write_source_csv(self, path):
        with path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.fields,
            )

            writer.writeheader()
            writer.writerows(self.sample_rows)

    def test_template_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            source = temp_dir / "source.csv"
            output = temp_dir / "annotations.csv"

            self.write_source_csv(source)

            count = create_relevance_template(
                source,
                output,
            )

            self.assertEqual(count, 2)
            self.assertTrue(output.exists())

            with output.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                rows = list(
                    csv.DictReader(file)
                )

            self.assertEqual(len(rows), 2)

            self.assertIn(
                "relevance_grade",
                rows[0],
            )

            self.assertIn(
                "review_note",
                rows[0],
            )

            self.assertIn(
                "review_confidence",
                rows[0],
            )

            self.assertEqual(
                rows[0]["relevance_grade"],
                "",
            )

            self.assertEqual(
                rows[0]["review_note"],
                "",
            )

            self.assertEqual(
                rows[0]["review_confidence"],
                "",
            )

    def test_existing_annotations_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            source = temp_dir / "source.csv"
            output = temp_dir / "annotations.csv"

            self.write_source_csv(source)

            output.write_text(
                "Existing annotation data",
                encoding="utf-8",
            )

            with self.assertRaises(
                FileExistsError
            ):
                create_relevance_template(
                    source,
                    output,
                )

            self.assertEqual(
                output.read_text(
                    encoding="utf-8"
                ),
                "Existing annotation data",
            )

    def test_missing_source_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            missing_source = (
                temp_dir / "missing.csv"
            )

            output = (
                temp_dir / "annotations.csv"
            )

            with self.assertRaises(
                FileNotFoundError
            ):
                create_relevance_template(
                    missing_source,
                    output,
                )

    def test_empty_source_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            source = temp_dir / "empty.csv"
            output = temp_dir / "annotations.csv"

            source.write_text(
                "",
                encoding="utf-8",
            )

            with self.assertRaises(
                ValueError
            ):
                create_relevance_template(
                    source,
                    output,
                )

    def test_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            source = temp_dir / "invalid.csv"
            output = temp_dir / "annotations.csv"

            source.write_text(
                "question,text\n"
                "What is glioma?,Passage\n",
                encoding="utf-8",
            )

            with self.assertRaises(
                ValueError
            ):
                create_relevance_template(
                    source,
                    output,
                )


if __name__ == "__main__":
    unittest.main()