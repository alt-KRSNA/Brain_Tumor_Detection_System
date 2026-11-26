
"""
Hybrid retrieval experiment.

Compares:
- Dense retrieval
- BM25 retrieval
- Hybrid retrieval (RRF)

Evaluation uses expected source matching and keyword coverage.
This is a preliminary evaluation, not a substitute for
passage-level human relevance judgments.
"""

import csv
import json
from pathlib import Path
from datetime import datetime

from genai.hybrid_retriever import HybridRetriever


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "hybrid_retrieval"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------

QUESTIONS = [
    {
        "question_id": "Q01",
        "question": "What are the characteristics of glioma?",
        "expected_sources": ["glioma.pdf"],
        "keywords": [
            "glioma",
            "tumor",
            "brain",
        ],
    },
    {
        "question_id": "Q02",
        "question": "What are the characteristics of meningioma?",
        "expected_sources": ["meningioma.pdf"],
        "keywords": [
            "meningioma",
            "tumor",
        ],
    },
    {
        "question_id": "Q03",
        "question": "What are the characteristics of pituitary tumors?",
        "expected_sources": [
            "pituitary_overview.pdf",
            "pituitary_diagnosis.pdf",
            "pituitary_symptoms.pdf",
            "pituitary_tumor.pdf",
        ],
        "keywords": [
            "pituitary",
            "tumor",
        ],
    },
    {
        "question_id": "Q04",
        "question": "What is a brain MRI used for?",
        "expected_sources": [
            "brain_mri_classification.pdf",
            "brain_tumor_overview.pdf",
        ],
        "keywords": [
            "MRI",
            "brain",
            "imaging",
        ],
    },
    {
        "question_id": "Q05",
        "question": "What are common symptoms of brain tumors?",
        "expected_sources": [
            "brain_tumor_overview.pdf",
            "glioma.pdf",
            "meningioma.pdf",
            "pituitary_symptoms.pdf",
        ],
        "keywords": [
            "headache",
            "seizure",
            "symptom",
        ],
    },
]


# ---------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------

def normalize_source(source: str) -> str:

    return Path(str(source)).name.lower().strip()


def source_match(
    metadata: dict,
    expected_sources: list[str],
) -> bool:

    actual_source = normalize_source(
        metadata.get("source", "")
    )

    expected = {
        normalize_source(source)
        for source in expected_sources
    }

    return actual_source in expected


def hit_at_k(
    results: list[dict],
    expected_sources: list[str],
    k: int,
) -> float:

    top_results = results[:k]

    return float(
        any(
            source_match(
                result.get("metadata", {}),
                expected_sources,
            )
            for result in top_results
        )
    )


def reciprocal_rank(
    results: list[dict],
    expected_sources: list[str],
) -> float:

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if source_match(
            result.get("metadata", {}),
            expected_sources,
        ):

            return 1.0 / rank

    return 0.0


def keyword_coverage(
    results: list[dict],
    keywords: list[str],
    k: int,
) -> float:

    if not keywords:
        return 0.0

    retrieved_text = " ".join(
        result.get("text", "")
        for result in results[:k]
    ).lower()

    matched = sum(
        1
        for keyword in keywords
        if keyword.lower() in retrieved_text
    )

    return matched / len(keywords)


def calculate_metrics(
    results: list[dict],
    expected_sources: list[str],
    keywords: list[str],
) -> dict:

    return {
        "Hit@3": hit_at_k(
            results,
            expected_sources,
            3,
        ),

        "Hit@5": hit_at_k(
            results,
            expected_sources,
            5,
        ),

        "MRR": reciprocal_rank(
            results,
            expected_sources,
        ),

        "Keyword_Coverage@3": keyword_coverage(
            results,
            keywords,
            3,
        ),

        "Keyword_Coverage@5": keyword_coverage(
            results,
            keywords,
            5,
        ),
    }


# ---------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------

def run_experiment():

    print("=" * 80)

    print("HYBRID RETRIEVAL EXPERIMENT")

    print("=" * 80)

    retriever = HybridRetriever(
        dense_top_k=10,
        bm25_top_k=10,
        final_top_k=5,
    )

    detailed_results = []

    metric_rows = []

    methods = [
        "dense_results",
        "bm25_results",
        "hybrid_results",
    ]

    for question_data in QUESTIONS:

        question_id = question_data["question_id"]

        question = question_data["question"]

        expected_sources = question_data["expected_sources"]

        keywords = question_data["keywords"]

        print("\n" + "-" * 80)

        print(f"{question_id}: {question}")

        print("-" * 80)

        retrieval_report = retriever.retrieve_with_details(
            query=question,
            final_top_k=5,
        )

        for method in methods:

            results = retrieval_report[method][:5]

            metrics = calculate_metrics(
                results=results,
                expected_sources=expected_sources,
                keywords=keywords,
            )

            metric_row = {
                "question_id": question_id,
                "question": question,
                "method": method.replace("_results", ""),
                **metrics,
            }

            metric_rows.append(metric_row)

            print(f"\nMethod: {method}")

            print(
                f"Hit@3: {metrics['Hit@3']:.2f}"
                f" | Hit@5: {metrics['Hit@5']:.2f}"
                f" | MRR: {metrics['MRR']:.2f}"
                f" | Keyword@3: "
                f"{metrics['Keyword_Coverage@3']:.2f}"
            )

            for rank, result in enumerate(
                results,
                start=1,
            ):

                metadata = result.get(
                    "metadata",
                    {},
                )

                print(
                    f"{rank}. "
                    f"{metadata.get('source', 'Unknown')} "
                    f"| Page {metadata.get('page', '?')}"
                )

        # Save passage details for this question.
        detailed_results.append({
            "question_id": question_id,
            "question": question,
            "expected_sources": expected_sources,
            "keywords": keywords,
            "retrieval": retrieval_report,
        })

    # -----------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------

    aggregate_metrics = {}

    for method in [
        "dense",
        "bm25",
        "hybrid",
    ]:

        rows = [
            row
            for row in metric_rows
            if row["method"] == method
        ]

        metric_names = [
            "Hit@3",
            "Hit@5",
            "MRR",
            "Keyword_Coverage@3",
            "Keyword_Coverage@5",
        ]

        aggregate_metrics[method] = {}

        for metric_name in metric_names:

            values = [
                row[metric_name]
                for row in rows
            ]

            aggregate_metrics[method][metric_name] = (
                sum(values) / len(values)
                if values
                else 0.0
            )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print("\n" + "=" * 80)

    print("AGGREGATE RESULTS")

    print("=" * 80)

    for method, metrics in aggregate_metrics.items():

        print(f"\n{method.upper()}")

        for metric_name, value in metrics.items():

            print(
                f"{metric_name:<25}: {value:.4f}"
            )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    csv_path = REPORT_DIR / "hybrid_metrics.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = list(metric_rows[0].keys())

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(metric_rows)

    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

    json_path = REPORT_DIR / "hybrid_retrieval_results.json"

    report = {
        "experiment": "Dense vs BM25 vs Hybrid RRF",
        "timestamp": datetime.now().isoformat(),
        "collection": retriever.collection_name,
        "corpus_size": len(retriever.documents),
        "dense_top_k": retriever.dense_top_k,
        "bm25_top_k": retriever.bm25_top_k,
        "final_top_k": retriever.final_top_k,
        "rrf_k": 60,
        "questions": detailed_results,
        "aggregate_metrics": aggregate_metrics,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # Save Markdown summary
    # -----------------------------------------------------

    markdown_path = REPORT_DIR / "hybrid_summary.md"

    with open(
        markdown_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write("# Hybrid Retrieval Experiment\n\n")

        file.write(
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        file.write(
            f"Collection: `{retriever.collection_name}`\n\n"
        )

        file.write(
            f"Corpus size: {len(retriever.documents)} chunks\n\n"
        )

        file.write(
            "## Aggregate Metrics\n\n"
        )

        file.write(
            "| Method | Hit@3 | Hit@5 | MRR | Keyword@3 | Keyword@5 |\n"
        )

        file.write(
            "|---|---:|---:|---:|---:|---:|\n"
        )

        for method, metrics in aggregate_metrics.items():

            file.write(
                f"| {method} "
                f"| {metrics['Hit@3']:.4f} "
                f"| {metrics['Hit@5']:.4f} "
                f"| {metrics['MRR']:.4f} "
                f"| {metrics['Keyword_Coverage@3']:.4f} "
                f"| {metrics['Keyword_Coverage@5']:.4f} |\n"
            )

        file.write(
            "\n## Interpretation Notes\n\n"
        )

        file.write(
            "- Hit@K checks whether an expected source appears in the top K.\n"
        )

        file.write(
            "- MRR uses the first expected-source match.\n"
        )

        file.write(
            "- Keyword coverage measures the presence of defined keywords.\n"
        )

        file.write(
            "- These metrics do not establish passage-level relevance.\n"
        )

        file.write(
            "- Five questions are insufficient for strong general conclusions.\n"
        )

    print("\n" + "=" * 80)

    print("FILES GENERATED")

    print("=" * 80)

    print(f"CSV     : {csv_path}")

    print(f"JSON    : {json_path}")

    print(f"Markdown: {markdown_path}")

    print("\nExperiment completed.")


if __name__ == "__main__":
    run_experiment()
