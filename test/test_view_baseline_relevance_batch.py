import csv
import tempfile
import unittest

from pathlib import Path

from test.view_baseline_relevance_batch import (
    load_annotation_records,
    get_batch,
    display_batch,
)


class TestBaselineRelevanceBatch(unittest.TestCase):

    def setUp(self):
        self.sample_rows = [
            {
                "configuration": "A - Baseline",
                "question_id": "Q01",
                "question": "What is glioma?",
                "rank": "2",
                "chunk_id": "chunk_2",
                "source": "glioma.pdf",
                "page": "5",
                "distance": "0.35",
                "text": "Second passage.",
                "relevance_grade": "",
                "review_note": "",
                "review_confidence": "",
            },
            {
                "configuration": "A - Baseline",
                "question_id": "Q01",
                "question": "What is glioma?",
                "rank": "1",
                "chunk_id": "chunk_1",
                "source": "glioma.pdf",
                "page": "4",
                "distance": "0.34",
                "text": "First passage.",
                "relevance_grade": "",
                "review_note": "",
                "review_confidence": "",
            },
            {
                "configuration": "A - Baseline",
                "question_id": "Q02",
                "question": "What is meningioma?",
                "rank": "1",
                "chunk_id": "chunk_3",
                "source": "meningioma.pdf",
                "page": "2",
                "distance": "0.30",
                "text": "Meningioma passage.",
                "relevance_grade": "2",
                "review_note": "Direct answer.",
                "review_confidence": "high",
            },
        ]

        self.fields = list(
            self.sample_rows[0].keys()
        )

    def write_csv(self, path):
        with path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.fields,
            )

            writer.writeheader()
            writer.writerows(self.sample_rows)

    def test_load_and_group_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.csv"
            self.write_csv(path)

            groups = load_annotation_records(path)

            self.assertEqual(len(groups), 2)
            self.assertEqual(len(groups[0]), 2)
            self.assertEqual(len(groups[1]), 1)

    def test_passages_are_sorted_by_rank(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.csv"
            self.write_csv(path)

            groups = load_annotation_records(path)

            self.assertEqual(
                groups[0][0]["rank"],
                "1",
            )

            self.assertEqual(
                groups[0][1]["rank"],
                "2",
            )

    def test_get_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.csv"
            self.write_csv(path)

            groups = load_annotation_records(path)

            batch_1 = get_batch(groups, 1)
            batch_2 = get_batch(groups, 2)

            self.assertEqual(
                batch_1[0]["question_id"],
                "Q01",
            )

            self.assertEqual(
                batch_2[0]["question_id"],
                "Q02",
            )

    def test_invalid_batch_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.csv"
            self.write_csv(path)

            groups = load_annotation_records(path)

            with self.assertRaises(ValueError):
                get_batch(groups, 0)

            with self.assertRaises(ValueError):
                get_batch(groups, 3)

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.csv"

            with self.assertRaises(FileNotFoundError):
                load_annotation_records(path)

    def test_display_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.csv"
            self.write_csv(path)

            groups = load_annotation_records(path)
            batch = get_batch(groups, 1)

            # Confirm display accepts the batch.
            display_batch(batch, 1)


if __name__ == "__main__":
    unittest.main()