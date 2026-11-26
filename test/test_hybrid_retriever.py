
"""
Unit tests for hybrid retrieval.

Run:
    python -m unittest test.test_hybrid_retriever -v
"""

import unittest

from genai.hybrid_retriever import (
    BM25,
    tokenize,
    reciprocal_rank_fusion,
)


class TestTokenizer(unittest.TestCase):

    def test_lowercase_tokenization(self):

        result = tokenize(
            "Brain MRI Detection"
        )

        self.assertEqual(
            result,
            ["brain", "mri", "detection"],
        )

    def test_hyphenated_words_are_preserved(self):

        result = tokenize(
            "brain-tumor MRI"
        )

        self.assertEqual(
            result,
            ["brain-tumor", "mri"],
        )

    def test_empty_text(self):

        self.assertEqual(
            tokenize(""),
            [],
        )


class TestBM25(unittest.TestCase):

    def setUp(self):

        self.documents = [
            "Glioma is a type of brain tumor.",
            "Meningioma develops from meninges.",
            "Pituitary tumors affect the pituitary gland.",
            "MRI imaging can help detect brain tumors.",
        ]

        self.bm25 = BM25(
            self.documents
        )

    def test_matching_document_scores_higher(self):

        scores = self.bm25.score(
            "glioma brain tumor"
        )

        self.assertGreater(
            scores[0],
            scores[1],
        )

    def test_retrieval_returns_matching_document(self):

        results = self.bm25.retrieve(
            "glioma",
            top_k=3,
        )

        self.assertGreater(
            len(results),
            0,
        )

        self.assertIn(
            "Glioma",
            results[0]["text"],
        )

    def test_unknown_query_returns_no_results(self):

        results = self.bm25.retrieve(
            "unrelated unknown vocabulary",
            top_k=3,
        )

        self.assertEqual(
            results,
            [],
        )

    def test_top_k_limit(self):

        results = self.bm25.retrieve(
            "brain tumor",
            top_k=2,
        )

        self.assertLessEqual(
            len(results),
            2,
        )

    def test_empty_corpus(self):

        bm25 = BM25([])

        self.assertEqual(
            bm25.score("glioma"),
            [],
        )

        self.assertEqual(
            bm25.retrieve("glioma"),
            [],
        )


class TestReciprocalRankFusion(unittest.TestCase):

    def test_document_in_both_rankings_scores_higher(self):

        dense_results = [
            {
                "id": "A",
                "text": "Document A",
                "metadata": {},
                "dense_score": 0.1,
            },
            {
                "id": "B",
                "text": "Document B",
                "metadata": {},
                "dense_score": 0.2,
            },
        ]

        bm25_results = [
            {
                "id": "B",
                "text": "Document B",
                "metadata": {},
                "bm25_score": 5.0,
            },
            {
                "id": "C",
                "text": "Document C",
                "metadata": {},
                "bm25_score": 3.0,
            },
        ]

        results = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
        )

        self.assertEqual(
            results[0]["id"],
            "B",
        )

    def test_union_of_rankings_is_preserved(self):

        dense_results = [
            {
                "id": "A",
                "text": "Document A",
                "metadata": {},
            },
        ]

        bm25_results = [
            {
                "id": "B",
                "text": "Document B",
                "metadata": {},
            },
        ]

        results = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
        )

        result_ids = {
            result["id"]
            for result in results
        }

        self.assertEqual(
            result_ids,
            {"A", "B"},
        )

    def test_empty_rankings(self):

        results = reciprocal_rank_fusion(
            [],
            [],
        )

        self.assertEqual(
            results,
            [],
        )

    def test_rank_metadata_is_preserved(self):

        dense_results = [
            {
                "id": "A",
                "text": "Document A",
                "metadata": {},
                "dense_score": 0.25,
            },
        ]

        bm25_results = [
            {
                "id": "A",
                "text": "Document A",
                "metadata": {},
                "bm25_score": 4.5,
            },
        ]

        results = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
        )

        self.assertEqual(
            results[0]["dense_rank"],
            1,
        )

        self.assertEqual(
            results[0]["bm25_rank"],
            1,
        )

        self.assertIsNotNone(
            results[0]["rrf_score"],
        )


if __name__ == "__main__":

    unittest.main()
