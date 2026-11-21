"""
Observed aggregate retrieval experiment results.

IMPORTANT:
These results were produced using provisional relevance judgments.
They are not final human-validated results.
"""

CONFIGURATION_RESULTS = {
    "A": {
        "chunk_size": 1000,
        "overlap": 200,
        "mrr": 1.0000,
        "precision_at_3": 0.9333,
        "recall_at_3": 0.7033,
        "ndcg_at_3": 0.8808,
        "avg_relevance_at_3": 1.3333,
        "precision_at_5": 0.8000,
        "recall_at_5": 1.0000,
        "ndcg_at_5": 0.9505,
        "avg_relevance_at_5": 1.0800,
    },
    "B": {
        "chunk_size": 700,
        "overlap": 150,
        "mrr": 1.0000,
        "precision_at_3": 1.0000,
        "recall_at_3": 0.6600,
        "ndcg_at_3": 0.8788,
        "avg_relevance_at_3": 1.6000,
        "precision_at_5": 0.9200,
        "recall_at_5": 1.0000,
        "ndcg_at_5": 0.8935,
        "avg_relevance_at_5": 1.2800,
    },
    "C": {
        "chunk_size": 500,
        "overlap": 100,
        "mrr": 1.0000,
        "precision_at_3": 0.9333,
        "recall_at_3": 0.6433,
        "ndcg_at_3": 0.6293,
        "avg_relevance_at_3": 1.2000,
        "precision_at_5": 0.8800,
        "recall_at_5": 1.0000,
        "ndcg_at_5": 0.8125,
        "avg_relevance_at_5": 1.2400,
    },
}

ANNOTATION_STATUS = "provisional_model_assisted"
QUERY_COUNT = 5
PASSAGES_PER_QUERY = 5