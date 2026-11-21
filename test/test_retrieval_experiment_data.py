import unittest

from test.retrieval_experiment_data import (
    CONFIGURATION_RESULTS,
    QUERY_COUNT,
    PASSAGES_PER_QUERY,
    ANNOTATION_STATUS,
)


class TestRetrievalExperimentData(unittest.TestCase):

    def test_three_configurations_exist(self):
        self.assertEqual(
            set(CONFIGURATION_RESULTS.keys()),
            {"A", "B", "C"},
        )

    def test_chunk_configurations_are_correct(self):
        self.assertEqual(
            CONFIGURATION_RESULTS["A"]["chunk_size"],
            1000,
        )

        self.assertEqual(
            CONFIGURATION_RESULTS["B"]["chunk_size"],
            700,
        )

        self.assertEqual(
            CONFIGURATION_RESULTS["C"]["chunk_size"],
            500,
        )

    def test_metrics_exist_for_every_configuration(self):
        required_metrics = {
            "mrr",
            "precision_at_3",
            "recall_at_3",
            "ndcg_at_3",
            "precision_at_5",
            "recall_at_5",
            "ndcg_at_5",
            "avg_relevance_at_5",
        }

        for config, values in CONFIGURATION_RESULTS.items():
            with self.subTest(configuration=config):
                self.assertTrue(
                    required_metrics.issubset(values.keys())
                )

    def test_metric_values_are_in_valid_range(self):
        normalized_metrics = [
            "mrr",
            "precision_at_3",
            "recall_at_3",
            "ndcg_at_3",
            "precision_at_5",
            "recall_at_5",
            "ndcg_at_5",
        ]

        for config, values in CONFIGURATION_RESULTS.items():
            for metric in normalized_metrics:
                with self.subTest(
                    configuration=config,
                    metric=metric,
                ):
                    self.assertGreaterEqual(
                        values[metric],
                        0.0,
                    )

                    self.assertLessEqual(
                        values[metric],
                        1.0,
                    )

    def test_evaluation_size(self):
        self.assertEqual(
            QUERY_COUNT * PASSAGES_PER_QUERY,
            25,
        )

    def test_annotation_status_is_provisional(self):
        self.assertEqual(
            ANNOTATION_STATUS,
            "provisional_model_assisted",
        )

    def test_config_b_has_best_precision_at_5(self):
        best_config = max(
            CONFIGURATION_RESULTS,
            key=lambda config: (
                CONFIGURATION_RESULTS[config]["precision_at_5"]
            ),
        )

        self.assertEqual(best_config, "B")

    def test_config_a_has_best_ndcg_at_5(self):
        best_config = max(
            CONFIGURATION_RESULTS,
            key=lambda config: (
                CONFIGURATION_RESULTS[config]["ndcg_at_5"]
            ),
        )

        self.assertEqual(best_config, "A")


if __name__ == "__main__":
    unittest.main()