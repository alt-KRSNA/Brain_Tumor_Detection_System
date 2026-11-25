# test/test_retrieval_baseline_inspection.py

import csv
import tempfile
import unittest
from pathlib import Path

from test.inspect_retrieval_baseline import (
    get_collection_name,
    summarize_question_records,
    build_question_summary,
    write_passage_csv,
    write_summary_csv,
)


class TestRetrievalBaselineInspection(unittest.TestCase):

    def setUp(self):
        self.sample_records = [
            {
                "configuration": "A - Baseline",
                "question_id": "Q01",
                "question": "What is a glioma brain tumor?",
                "expected_sources": ["glioma.pdf"],
                "passages": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk_1",
                        "source": "glioma.pdf",
                        "page": 1,
                        "distance": 0.15,
                        "text": "Glioma passage text.",
                    },
                    {
                        "rank": 2,
                        "chunk_id": "chunk_2",
                        "source": "meningioma.pdf",
                        "page": 3,
                        "distance": 0.30,
                        "text": "Meningioma passage text.",
                    },
                    {
                        "rank": 3,
                        "chunk_id": "chunk_3",
                        "source": "glioma.pdf",
                        "page": 2,
                        "distance": 0.35,
                        "text": "Another glioma passage.",
                    },
                ],
            },
            {
                "configuration": "A - Baseline",
                "question_id": "Q02",
                "question": "What are the symptoms of a brain tumor?",
                "expected_sources": [
                    "glioma.pdf",
                    "meningioma.pdf",
                ],
                "passages": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk_4",
                        "source": "pituitary_overview.pdf",
                        "page": 2,
                        "distance": 0.20,
                        "text": "Pituitary passage text.",
                    },
                    {
                        "rank": 2,
                        "chunk_id": "chunk_5",
                        "source": "brain_mri_classification.pdf",
                        "page": 5,
                        "distance": 0.25,
                        "text": "MRI passage text.",
                    },
                ],
            },
        ]

    def test_configuration_collection_names(self):
        config_a = {
            "name": "A - Baseline",
        }

        config_b = {
            "name": "B - Medium",
        }

        config_c = {
            "name": "C - Small",
        }

        self.assertEqual(
            get_collection_name(config_a),
            "experiment_chunking_a",
        )

        self.assertEqual(
            get_collection_name(config_b),
            "experiment_chunking_b",
        )

        self.assertEqual(
            get_collection_name(config_c),
            "experiment_chunking_c",
        )

    def test_summary_metrics(self):
        metrics = summarize_question_records(
            self.sample_records
        )

        self.assertEqual(
            metrics["total_questions"],
            2,
        )

        # Q01 has an expected source in the top 5.
        # Q02 has no expected source in the retrieved results.
        self.assertEqual(
            metrics["source_hit_at_5"],
            0.5,
        )

        # Rank 1 for Q01 is an expected source.
        # Rank 1 for Q02 is not.
        self.assertEqual(
            metrics["top1_expected_source_rate"],
            0.5,
        )

        # Q01: 2 matching sources in top 3.
        # Q02: 0 matching sources in top 2.
        self.assertEqual(
            metrics["average_expected_source_count_at_5"],
            1.0,
        )

        self.assertAlmostEqual(
            metrics["average_top1_distance"],
            0.175,
        )

    def test_empty_summary(self):
        metrics = summarize_question_records([])

        self.assertEqual(
            metrics["total_questions"],
            0,
        )

        self.assertEqual(
            metrics["source_hit_at_5"],
            0.0,
        )

        self.assertEqual(
            metrics["top1_expected_source_rate"],
            0.0,
        )

    def test_build_question_summary(self):
        record = self.sample_records[0]

        summary = build_question_summary(
            "A - Baseline",
            record,
        )

        self.assertEqual(
            summary["configuration"],
            "A - Baseline",
        )

        self.assertEqual(
            summary["top1_source"],
            "glioma.pdf",
        )

        self.assertTrue(
            summary["top1_expected_source"],
        )

        self.assertEqual(
            summary["first_expected_source_rank"],
            1,
        )

        self.assertEqual(
            summary["expected_source_passages_at_5"],
            2,
        )

    def test_passage_csv_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = (
                Path(temp_dir)
                / "passages.csv"
            )

            write_passage_csv(
                self.sample_records,
                output_path,
            )

            with output_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                rows = list(
                    csv.DictReader(file)
                )

            # Three passages from Q01 and two from Q02.
            self.assertEqual(
                len(rows),
                5,
            )

            self.assertEqual(
                rows[0]["source"],
                "glioma.pdf",
            )

            self.assertEqual(
                rows[0]["rank"],
                "1",
            )

            self.assertIn(
                "Glioma passage",
                rows[0]["text"],
            )

    def test_summary_csv_export(self):
        summary_rows = [
            build_question_summary(
                record["configuration"],
                record,
            )
            for record in self.sample_records
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = (
                Path(temp_dir)
                / "summary.csv"
            )

            write_summary_csv(
                summary_rows,
                output_path,
            )

            with output_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                rows = list(
                    csv.DictReader(file)
                )

            self.assertEqual(
                len(rows),
                2,
            )

            self.assertEqual(
                rows[0]["top1_source"],
                "glioma.pdf",
            )

            self.assertEqual(
                rows[1]["top1_expected_source"],
                "False",
            )


if __name__ == "__main__":
    unittest.main()