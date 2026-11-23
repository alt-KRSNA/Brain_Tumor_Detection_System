 
import unittest

from test.failure_investigator import analyze_retrieval_failures


class TestFailureInvestigator(unittest.TestCase):

    def setUp(self):
        self.records = [
            {
                "annotation_key": "A|glioma|1",
                "configuration": "A",
                "question_id": "glioma",
                "question": "What is glioma?",
                "rank": 1,
                "relevance_grade": 0,
                "chunk_id": "chunk_1",
                "source": "glioma.pdf",
                "page": 5,
                "distance": 0.36,
                "text": "Unrelated passage.",
            },
            {
                "annotation_key": "A|glioma|2",
                "configuration": "A",
                "question_id": "glioma",
                "question": "What is glioma?",
                "rank": 2,
                "relevance_grade": 1,
                "chunk_id": "chunk_2",
                "source": "glioma.pdf",
                "page": 6,
                "distance": 0.40,
                "text": "Partially relevant passage.",
            },
            {
                "annotation_key": "B|glioma|1",
                "configuration": "B",
                "question_id": "glioma",
                "question": "What is glioma?",
                "rank": 1,
                "relevance_grade": 2,
                "chunk_id": "chunk_1",
                "source": "glioma.pdf",
                "page": 5,
                "distance": 0.30,
                "text": "Directly relevant passage.",
            },
        ]

    def test_overall_grade_counts(self):
        report = analyze_retrieval_failures(self.records)

        self.assertEqual(report["overall"]["total"], 3)
        self.assertEqual(report["overall"]["grade_0"], 1)
        self.assertEqual(report["overall"]["grade_1"], 1)
        self.assertEqual(report["overall"]["grade_2"], 1)

    def test_failure_rate(self):
        report = analyze_retrieval_failures(self.records)

        self.assertEqual(report["failure_count"], 2)
        self.assertAlmostEqual(report["failure_rate"], 2 / 3, places=4)

    def test_failure_types_are_distinguished(self):
        report = analyze_retrieval_failures(self.records)

        types = {
            record["failure_type"]
            for record in report["failure_details"]
        }

        self.assertEqual(
            types,
            {"irrelevant", "partially_relevant"},
        )

    def test_repeated_chunks_are_detected(self):
        report = analyze_retrieval_failures(self.records)

        repeated = report["repeated_chunks"]

        self.assertEqual(len(repeated), 1)
        self.assertEqual(repeated[0]["chunk_id"], "chunk_1")
        self.assertEqual(
            repeated[0]["configurations"],
            ["A", "B"],
        )

    def test_empty_records_raise_error(self):
        with self.assertRaises(ValueError):
            analyze_retrieval_failures([])


if __name__ == "__main__":
    unittest.main()
