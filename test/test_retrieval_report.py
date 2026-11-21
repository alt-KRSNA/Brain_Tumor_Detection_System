import unittest

from test.generate_retrieval_report import (
    get_rankings,
    generate_markdown_report,
    export_comparison_csv,
)


class TestRetrievalReport(unittest.TestCase):

    def test_ndcg_ranking(self):
        ranking = get_rankings("ndcg_at_5")

        self.assertEqual(
            [config for config, _ in ranking],
            ["A", "B", "C"],
        )

    def test_precision_ranking(self):
        ranking = get_rankings("precision_at_5")

        self.assertEqual(
            [config for config, _ in ranking],
            ["B", "C", "A"],
        )

    def test_report_generation(self):
        report = generate_markdown_report()

        self.assertTrue(report.exists())

        content = report.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "Retrieval Chunking Experiment Report",
            content,
        )

        self.assertIn("Config A", content)
        self.assertIn("Config B", content)
        self.assertIn("Config C", content)

    def test_csv_generation(self):
        csv_file = export_comparison_csv()

        self.assertTrue(csv_file.exists())

        content = csv_file.read_text(
            encoding="utf-8"
        )

        self.assertIn("configuration", content)
        self.assertIn("precision_at_5", content)
        self.assertIn("ndcg_at_5", content)


if __name__ == "__main__":
    unittest.main()