import unittest

from genai.context_builder import ContextBuilder


class TestContextBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = ContextBuilder(
            max_context_tokens=100,
            max_passage_tokens=30,
        )

    def test_duplicate_passages_are_removed(self):
        passages = [
            {"text": "Glioma is a brain tumor.", "score": 0.9},
            {"text": " glioma IS a brain tumor. ", "score": 0.8},
        ]

        result = self.builder.build_context(passages)

        self.assertEqual(result["passage_count"], 1)

    def test_empty_passages_are_ignored(self):
        passages = [
            {"text": ""},
            {"text": "   "},
            {"text": "Useful medical information."},
        ]

        result = self.builder.build_context(passages)

        self.assertEqual(result["passage_count"], 1)

    def test_relevance_ordering(self):
        passages = [
            {"text": "Low relevance passage.", "score": 0.2},
            {"text": "High relevance passage.", "score": 0.9},
        ]

        result = self.builder.build_context(passages)

        self.assertEqual(
            result["passages"][0]["text"],
            "High relevance passage.",
        )

    def test_context_budget_is_respected(self):
        builder = ContextBuilder(
            max_context_tokens=10,
            max_passage_tokens=10,
        )

        passages = [
            {"text": "one two three four five", "score": 0.9},
            {"text": "six seven eight nine ten", "score": 0.8},
            {"text": "eleven twelve thirteen", "score": 0.7},
        ]

        result = builder.build_context(passages)

        self.assertLessEqual(
            result["estimated_tokens"],
            builder.max_context_tokens,
        )

    def test_passage_length_is_limited(self):
        builder = ContextBuilder(
            max_context_tokens=100,
            max_passage_tokens=5,
        )

        passages = [
            {
                "text": "one two three four five six seven eight nine ten",
                "score": 1.0,
            }
        ]

        result = builder.build_context(passages)

        self.assertEqual(
            result["passages"][0]["text"],
            "one two three four five",
        )

    def test_source_metadata_is_preserved(self):
        passages = [
            {
                "text": "Pituitary tumor information.",
                "score": 0.8,
                "metadata": {
                    "source": "pituitary_overview.pdf",
                    "page": 5,
                },
            }
        ]

        result = self.builder.build_context(passages)

        self.assertEqual(
            result["passages"][0]["source"],
            "pituitary_overview.pdf",
        )

        self.assertEqual(
            result["passages"][0]["page"],
            5,
        )

    def test_citations_are_in_context(self):
        passages = [
            {
                "text": "Meningiomas arise from meninges.",
                "score": 0.8,
                "metadata": {
                    "source": "meningioma.pdf",
                    "page": 12,
                },
            }
        ]

        result = self.builder.build_context(passages)

        self.assertIn(
            "[Source: meningioma.pdf, Page: 12]",
            result["context"],
        )

    def test_empty_input(self):
        result = self.builder.build_context([])

        self.assertEqual(result["passage_count"], 0)
        self.assertEqual(result["context"], "")
        self.assertEqual(result["sources"], [])

    def test_source_list_does_not_duplicate_references(self):
        passages = [
            {
                "text": "First passage.",
                "score": 0.9,
                "metadata": {"source": "brain.pdf", "page": 2},
            },
            {
                "text": "Second passage.",
                "score": 0.8,
                "metadata": {"source": "brain.pdf", "page": 2},
            },
        ]

        result = self.builder.build_context(passages)

        self.assertEqual(len(result["sources"]), 1)


if __name__ == "__main__":
    unittest.main()