
import json
import tempfile
import unittest
from pathlib import Path

from test.retrieval_evidence import (
    RetrievalEvidenceError,
    load_retrieval_evidence,
    validate_annotations,
)


class TestRetrievalEvidence(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.candidates_path = self.root / "candidates.json"
        self.annotations_path = self.root / "annotations.json"

        candidates = {
            "configurations": {
                "A": {
                    "glioma": {
                        "question": "What is glioma?",
                        "passages": [
                            {
                                "chunk_id": "chunk_1",
                                "rank": 1,
                                "source": "glioma.pdf",
                                "page": 5,
                                "distance": 0.36,
                                "text": "Example passage text.",
                                "relevance_grade": None,
                            }
                        ],
                    }
                }
            }
        }

        annotations = {
            "A|glioma|1": {
                "configuration": "A",
                "question_id": "glioma",
                "rank": 1,
                "relevance_grade": 2,
            }
        }

        self.candidates_path.write_text(
            json.dumps(candidates),
            encoding="utf-8",
        )

        self.annotations_path.write_text(
            json.dumps(annotations),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loads_and_joins_evidence(self):
        records = load_retrieval_evidence(
            self.candidates_path,
            self.annotations_path,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["relevance_grade"], 2)
        self.assertEqual(records[0]["chunk_id"], "chunk_1")
        self.assertEqual(records[0]["text"], "Example passage text.")

    def test_missing_annotation_raises_error(self):
        self.annotations_path.write_text(
            json.dumps({}),
            encoding="utf-8",
        )

        with self.assertRaises(RetrievalEvidenceError):
            load_retrieval_evidence(
                self.candidates_path,
                self.annotations_path,
            )

    def test_annotation_key_mismatch_raises_error(self):
        annotations = {
            "A|glioma|2": {
                "configuration": "A",
                "question_id": "glioma",
                "rank": 1,
                "relevance_grade": 2,
            }
        }

        with self.assertRaises(RetrievalEvidenceError):
            validate_annotations(annotations)

    def test_invalid_grade_raises_error(self):
        annotations = {
            "A|glioma|1": {
                "configuration": "A",
                "question_id": "glioma",
                "rank": 1,
                "relevance_grade": 3,
            }
        }

        with self.assertRaises(RetrievalEvidenceError):
            validate_annotations(annotations)


if __name__ == "__main__":
    unittest.main()
