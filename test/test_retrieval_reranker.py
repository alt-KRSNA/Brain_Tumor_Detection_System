import unittest

from genai.retrieval_reranker import RetrievalReranker


class TestRetrievalReranker(unittest.TestCase):

    def setUp(self):
        self.reranker = RetrievalReranker()

    def test_empty_passages(self):
        result = self.reranker.rerank(
            "What is glioma?",
            [],
        )
        self.assertEqual(result, [])

    def test_relevant_passage_ranks_higher(self):
        passages = [
            {"text": "General information about MRI scans."},
            {"text": "Glioma is a tumor arising from glial cells."},
            {"text": "Meningiomas originate from meninges."},
        ]

        result = self.reranker.rerank(
            "Explain glioma tumor",
            passages,
        )

        self.assertEqual(
            result[0]["text"],
            "Glioma is a tumor arising from glial cells.",
        )

    def test_lexical_score(self):
        query_terms = {"glioma", "tumor"}

        score = self.reranker.calculate_lexical_score(
            query_terms,
            "Glioma is a brain tumor.",
        )

        self.assertEqual(score, 1.0)

    def test_partial_match(self):
        query_terms = {"glioma", "tumor", "symptoms"}

        score = self.reranker.calculate_lexical_score(
            query_terms,
            "Glioma is a brain tumor.",
        )

        self.assertAlmostEqual(score, 2 / 3)

    def test_no_match(self):
        score = self.reranker.calculate_lexical_score(
            {"glioma", "tumor"},
            "MRI scans help visualize the brain.",
        )

        self.assertEqual(score, 0.0)

    def test_original_passage_fields_are_preserved(self):
        passage = {
            "text": "Glioma is a brain tumor.",
            "score": 0.85,
            "metadata": {"source": "glioma.pdf"},
        }

        result = self.reranker.rerank(
            "glioma tumor",
            [passage],
        )

        self.assertEqual(result[0]["score"], 0.85)
        self.assertEqual(
            result[0]["metadata"]["source"],
            "glioma.pdf",
        )

    def test_reranking_does_not_modify_original(self):
        passages = [
            {"text": "Glioma tumor"},
            {"text": "MRI scan"},
        ]

        original = [passage.copy() for passage in passages]

        self.reranker.rerank("glioma tumor", passages)

        self.assertEqual(passages, original)

    def test_original_rank_is_recorded(self):
        passages = [
            {"text": "MRI scan"},
            {"text": "Glioma tumor"},
        ]

        result = self.reranker.rerank(
            "glioma tumor",
            passages,
        )

        glioma = next(
            item for item in result
            if item["text"] == "Glioma tumor"
        )

        self.assertEqual(glioma["original_rank"], 2)

    def test_stopwords_are_removed(self):
        terms = self.reranker.extract_query_terms(
            "What is the glioma tumor?"
        )

        self.assertEqual(terms, {"glioma", "tumor"})


if __name__ == "__main__":
    unittest.main()
    