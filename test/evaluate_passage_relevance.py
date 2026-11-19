import json
import math
import csv
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANNOTATION_FILE = PROJECT_ROOT / "test" / "passage_annotations.json"

RESULTS_CSV = PROJECT_ROOT / "test" / "passage_evaluation_results.csv"
SUMMARY_JSON = PROJECT_ROOT / "test" / "passage_evaluation_summary.json"

CONFIGURATIONS = ["A", "B", "C"]

QUESTION_IDS = [
    "glioma",
    "meningioma",
    "pituitary",
    "mri",
    "symptoms",
]

K_VALUES = [3, 5]


def load_annotations():
    """Load and validate passage relevance annotations."""

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_FILE}"
        )

    with ANNOTATION_FILE.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    if not isinstance(annotations, dict):
        raise ValueError("Annotations must be a JSON dictionary.")

    return annotations


def organize_annotations(annotations):
    """Organize annotations by configuration and question."""

    organized = defaultdict(
        lambda: defaultdict(dict)
    )

    for key, entry in annotations.items():

        if not isinstance(entry, dict):
            continue

        config = entry.get("configuration")
        question_id = entry.get("question_id")
        rank = entry.get("rank")
        grade = entry.get("relevance_grade")

        if config not in CONFIGURATIONS:
            continue

        if question_id not in QUESTION_IDS:
            continue

        if grade not in [0, 1, 2]:
            raise ValueError(
                f"Invalid relevance grade for {key}: {grade}"
            )

        if not isinstance(rank, int) or rank < 1:
            raise ValueError(
                f"Invalid rank for annotation {key}: {rank}"
            )

        organized[config][question_id][rank] = grade

    return organized


def validate_annotations(organized):
    """Ensure all expected passages have been annotated."""

    missing = []

    for config in CONFIGURATIONS:
        for question_id in QUESTION_IDS:

            grades = organized.get(config, {}).get(question_id, {})

            for rank in range(1, 6):

                if rank not in grades:
                    missing.append(
                        f"{config}|{question_id}|rank {rank}"
                    )

    if missing:

        print("\nANNOTATION VALIDATION FAILED")
        print("=" * 70)
        print(f"Missing annotations: {len(missing)}")

        for item in missing:
            print(" -", item)

        raise ValueError(
            "\nComplete all 75 passage annotations before evaluation."
        )

    print("Annotation validation passed: all 75 passages are graded.")


def precision_at_k(grades, k):
    """Fraction of top-k passages that are relevant (grade >= 1)."""

    top_k = grades[:k]

    if not top_k:
        return 0.0

    relevant = sum(grade >= 1 for grade in top_k)

    return relevant / len(top_k)


def recall_at_k(grades, k):
    """
    Pool-relative Recall@K.

    The denominator is the total number of relevant passages
    among the five manually annotated retrieved passages.
    """

    total_relevant = sum(grade >= 1 for grade in grades)

    if total_relevant == 0:
        return 0.0

    retrieved_relevant = sum(
        grade >= 1 for grade in grades[:k]
    )

    return retrieved_relevant / total_relevant


def mean_reciprocal_rank(grades):
    """Reciprocal rank of the first relevant passage."""

    for rank, grade in enumerate(grades, start=1):

        if grade >= 1:
            return 1.0 / rank

    return 0.0


def dcg_at_k(grades, k):
    """Calculate discounted cumulative gain."""

    score = 0.0

    for rank, grade in enumerate(grades[:k], start=1):

        gain = (2 ** grade) - 1

        score += gain / math.log2(rank + 1)

    return score


def ndcg_at_k(grades, k):
    """Calculate normalized discounted cumulative gain."""

    actual_dcg = dcg_at_k(grades, k)

    ideal_grades = sorted(
        grades,
        reverse=True
    )

    ideal_dcg = dcg_at_k(ideal_grades, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def average_relevance_at_k(grades, k):
    """Mean relevance grade among top-k passages."""

    top_k = grades[:k]

    if not top_k:
        return 0.0

    return sum(top_k) / len(top_k)


def evaluate_question(grades):
    """Calculate retrieval metrics for a single question."""

    metrics = {
        "MRR": mean_reciprocal_rank(grades)
    }

    for k in K_VALUES:

        metrics[f"Precision@{k}"] = precision_at_k(
            grades, k
        )

        metrics[f"Recall@{k}"] = recall_at_k(
            grades, k
        )

        metrics[f"nDCG@{k}"] = ndcg_at_k(
            grades, k
        )

        metrics[f"AvgRelevance@{k}"] = (
            average_relevance_at_k(grades, k)
        )

    return metrics


def calculate_configuration_summary(question_results):
    """Calculate mean metrics across all questions."""

    metric_names = list(
        question_results[0]["metrics"].keys()
    )

    summary = {}

    for metric_name in metric_names:

        values = [
            result["metrics"][metric_name]
            for result in question_results
        ]

        summary[metric_name] = sum(values) / len(values)

    return summary


def save_csv_results(results):
    """Save question-level metrics to CSV."""

    if not results:
        return

    fieldnames = [
        "configuration",
        "question_id",
        "MRR",
        "Precision@3",
        "Recall@3",
        "nDCG@3",
        "AvgRelevance@3",
        "Precision@5",
        "Recall@5",
        "nDCG@5",
        "AvgRelevance@5",
    ]

    with RESULTS_CSV.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            row = {
                "configuration": result["configuration"],
                "question_id": result["question_id"],
            }

            row.update(result["metrics"])

            writer.writerow(row)

    print(f"\nDetailed results saved: {RESULTS_CSV}")


def save_summary(summary):
    """Save configuration-level averages to JSON."""

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(f"Summary saved: {SUMMARY_JSON}")


def print_results(results, summary):
    """Print question-level and configuration-level metrics."""

    print("\n" + "=" * 90)
    print("PASSAGE-LEVEL RETRIEVAL EVALUATION")
    print("=" * 90)

    for config in CONFIGURATIONS:

        print(f"\nCONFIGURATION {config}")
        print("-" * 90)

        config_results = [
            result
            for result in results
            if result["configuration"] == config
        ]

        for result in config_results:

            metrics = result["metrics"]

            print(f"\nQuestion: {result['question_id']}")

            print(
                f"MRR: {metrics['MRR']:.4f} | "
                f"P@3: {metrics['Precision@3']:.4f} | "
                f"R@3: {metrics['Recall@3']:.4f} | "
                f"nDCG@3: {metrics['nDCG@3']:.4f}"
            )

            print(
                f"P@5: {metrics['Precision@5']:.4f} | "
                f"R@5: {metrics['Recall@5']:.4f} | "
                f"nDCG@5: {metrics['nDCG@5']:.4f} | "
                f"AvgRel@5: {metrics['AvgRelevance@5']:.4f}"
            )

        print(f"\nCONFIGURATION {config} AVERAGE")

        for metric, value in summary[config].items():

            print(f"{metric:<20}: {value:.4f}")


def main():

    print("=" * 70)
    print("PASSAGE RELEVANCE EVALUATOR")
    print("=" * 70)

    annotations = load_annotations()

    organized = organize_annotations(
        annotations
    )

    validate_annotations(
        organized
    )

    all_results = []
    configuration_summaries = {}

    for config in CONFIGURATIONS:

        question_results = []

        for question_id in QUESTION_IDS:

            grades = [
                organized[config][question_id][rank]
                for rank in range(1, 6)
            ]

            metrics = evaluate_question(
                grades
            )

            result = {
                "configuration": config,
                "question_id": question_id,
                "grades": grades,
                "metrics": metrics,
            }

            question_results.append(result)
            all_results.append(result)

        configuration_summaries[config] = (
            calculate_configuration_summary(
                question_results
            )
        )

    print_results(
        all_results,
        configuration_summaries
    )

    save_csv_results(
        all_results
    )

    save_summary(
        configuration_summaries
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()