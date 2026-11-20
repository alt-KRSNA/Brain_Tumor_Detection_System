import unittest

from genai.response_validator import ResponseValidator


class TestResponseValidator(unittest.TestCase):

    def setUp(self):
        self.validator = ResponseValidator()

        self.sources = [
            {"source": "glioma.pdf", "page": 5},
            {"source": "meningioma.pdf", "page": 12},
        ]

    def test_valid_citation(self):
        answer = (
            "Gliomas arise from glial cells. "
            "[Source: glioma.pdf, Page: 5]"
        )

        result = self.validator.validate(
            answer,
            self.sources,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["status"], "valid")

    def test_missing_citations(self):
        answer = "Gliomas arise from glial cells."

        result = self.validator.validate(
            answer,
            self.sources,
        )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["status"],
            "missing_citations",
        )

    def test_wrong_source(self):
        answer = (
            "Glioma information. "
            "[Source: unknown.pdf, Page: 5]"
        )

        result = self.validator.validate(
            answer,
            self.sources,
        )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["status"],
            "unsupported_citations",
        )

    def test_wrong_page(self):
        answer = (
            "Glioma information. "
            "[Source: glioma.pdf, Page: 99]"
        )

        result = self.validator.validate(
            answer,
            self.sources,
        )

        self.assertFalse(result["valid"])

    def test_multiple_valid_citations(self):
        answer = (
            "Information about glioma. "
            "[Source: glioma.pdf, Page: 5] "
            "Additional information. "
            "[Source: meningioma.pdf, Page: 12]"
        )

        result = self.validator.validate(
            answer,
            self.sources,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["citations"]), 2)

    def test_mixed_valid_and_invalid_citations(self):
        answer = (
            "[Source: glioma.pdf, Page: 5] "
            "[Source: fake.pdf, Page: 1]"
        )

        result = self.validator.validate(
            answer,
            self.sources,
        )

        self.assertFalse(result["valid"])
        self.assertEqual(
            len(result["unsupported_citations"]),
            1,
        )

    def test_source_case_insensitive(self):
        answer = (
            "[Source: GLIOMA.PDF, Page: 5]"
        )

        result = self.validator.validate(
            answer,
            self.sources,
        )

        self.assertTrue(result["valid"])

    def test_extract_citations(self):
        answer = (
            "Information "
            "[Source: glioma.pdf, Page: 5]"
        )

        citations = self.validator.extract_citations(answer)

        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["source"], "glioma.pdf")
        self.assertEqual(citations[0]["page"], 5)


if __name__ == "__main__":
    unittest.main()