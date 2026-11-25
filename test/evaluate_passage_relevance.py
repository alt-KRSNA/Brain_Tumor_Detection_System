
import csv
import json
import math
from pathlib import Path
from collections import defaultdict, Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANNOTATION_FILE = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "retrieval_baseline"
    / "baseline_relevance_annotations.csv"
)

RESULTS_CSV = PROJECT_ROOT / "test" / "passage_evaluation_results.csv"
SUMMARY_JSON = PROJECT_ROOT / "test" / "passage_evaluation_summary.json"

CONFIGURATIONS = {
    "A": "A - Baseline",
    "B": "B - Medium",
    "C": "C - Small",
}

QUESTION_IDS = ["Q01", "Q02", "Q03", "Q04", "Q05"]

K_VALUES = [3, 5]


# ============================================================
# 1. LOAD ANNOTATIONS
# ============================================================

def load_annotations():
    """Load human-annotated passages from CSV."""

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_FILE}"
        )

    with ANNOTATION_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "configuration",
            "question_id",
            "rank",
            "relevance_grade",
            "review_note",
            "review_confidence",
        }

        if not required_columns.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV is missing required columns.")

        annotations = list(reader)

    if len(annotations) != 75:
        raise ValueError(
            f"Expected 75 annotations, found {len(annotations)}."
        )

    return annotations


# ============================================================
# 2. ORGANIZE AND VALIDATE ANNOTATIONS
# ============================================================

def organize_annotations(annotations):
    """Organize relevance grades by configuration, question and rank."""

    organized = defaultdict(lambda: defaultdict(dict))

    reverse_config = {
        label: code
        for code, label in CONFIGURATIONS.items()
    }

    seen = set()

    for row in annotations:

        label = row["configuration"].strip()
        question_id = row["question_id"].strip()

        if label not in reverse_config:
            raise ValueError(
                f"Unknown configuration: {label}"
            )

        config = reverse_config[label]

        if question_id not in QUESTION_IDS:
            raise ValueError(
                f"Unknown question ID: {question_id}"
            )

        try:
            rank = int(row["rank"])
            grade = int(row["relevance_grade"])
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid rank or relevance grade: {row}"
            )

        if rank not in range(1, 6):
            raise ValueError(f"Invalid rank: {rank}")

        if grade not in (0, 1, 2):
            raise ValueError(f"Invalid relevance grade: {grade}")

        if not row["review_note"].strip():
            raise ValueError(
                f"Missing review note: {config}/{question_id}/{rank}"
            )

        if not row["review_confidence"].strip():
            raise ValueError(
                f"Missing review confidence: {config}/{question_id}/{rank}"
            )

        key = (config, question_id, rank)

        if key in seen:
            raise ValueError(
                f"Duplicate annotation: {key}"
            )

        seen.add(key)

        organized[config][question_id][rank] = grade

    if len(seen) != 75:
        raise ValueError(
            f"Expected 75 unique annotations, found {len(seen)}."
        )

    return organized


def validate_annotations(organized):
    """Ensure all 75 expected passages have been annotated."""

    missing = []

    for config in CONFIGURATIONS:

        for question_id in QUESTION_IDS:

            grades = organized.get(
                config, {}
            ).get(question_id, {})

            for rank in range(1, 6):

                if rank not in grades:
                    missing.append(
                        f"{config}|{question_id}|rank {rank}"
                    )

    if missing:

        print("\nANNOTATION VALIDATION FAILED")

        for item in missing:
            print(" -", item)

        raise ValueError(
            "Complete all 75 annotations before evaluation."
        )

    print("Annotation validation passed: all 75 passages are graded.")


# ============================================================
# 3. BASIC RETRIEVAL METRICS
# ============================================================

def precision_at_k(grades, k):
    """Fraction of top-K passages with relevance grade >= 1."""

    top_k = grades[:k]

    if not top_k:
        return 0.0

    relevant = sum(
        grade >= 1 for grade in top_k
    )

    return relevant / len(top_k)


def recall_at_k(grades, k):
    """
    Pool-relative Recall@K.

    Denominator: relevant passages among the five
    manually annotated retrieved passages.
    """

    total_relevant = sum(
        grade >= 1 for grade in grades
    )

    if total_relevant == 0:
        return 0.0

    retrieved_relevant = sum(
        grade >= 1 for grade in grades[:k]
    )

    return retrieved_relevant / total_relevant


def mean_reciprocal_rank(grades):
    """First passage with relevance grade >= 1."""

    for rank, grade in enumerate(grades, start=1):

        if grade >= 1:
            return 1.0 / rank

    return 0.0


# ============================================================
# 4. STRICT RETRIEVAL METRICS
# ============================================================

def strict_mrr(grades):
    """
    Reciprocal rank of the first directly useful passage.

    Only relevance grade 2 counts.
    """

    for rank, grade in enumerate(grades, start=1):

        if grade == 2:
            return 1.0 / rank

    return 0.0


def hit_rate_at_k(grades, k):
    """
    Returns 1 if at least one grade-2 passage
    appears in the top K, otherwise 0.
    """

    return float(
        any(grade == 2 for grade in grades[:k])
    )


# ============================================================
# 5. RANKING QUALITY METRICS
# ============================================================

def dcg_at_k(grades, k):
    """Discounted Cumulative Gain."""

    score = 0.0

    for rank, grade in enumerate(grades[:k], start=1):

        gain = (2 ** grade) - 1

        score += gain / math.log2(rank + 1)

    return score


def ndcg_at_k(grades, k):
    """Normalized Discounted Cumulative Gain."""

    actual_dcg = dcg_at_k(grades, k)

    ideal_grades = sorted(
        grades,
        reverse=True
    )

    ideal_dcg = dcg_at_k(
        ideal_grades, k
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def average_relevance_at_k(grades, k):
    """Average relevance grade among top-K passages."""

    top_k = grades[:k]

    if not top_k:
        return 0.0

    return sum(top_k) / len(top_k)


# ============================================================
# 6. QUESTION-LEVEL EVALUATION
# ============================================================

def evaluate_question(grades):
    """Calculate all retrieval metrics for one question."""

    metrics = {
        "MRR": mean_reciprocal_rank(grades),
        "Strict_MRR": strict_mrr(grades),
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

        metrics[f"HitRate@{k}"] = hit_rate_at_k(
            grades, k
        )

    return metrics


# ============================================================
# 7. CONFIGURATION-LEVEL SUMMARY
# ============================================================

def calculate_configuration_summary(question_results):
    """Calculate mean metrics across all five questions."""

    metric_names = list(
        question_results[0]["metrics"].keys()
    )

    summary = {}

    for metric_name in metric_names:

        values = [
            result["metrics"][metric_name]
            for result in question_results
        ]

        summary[metric_name] = (
            sum(values) / len(values)
        )

    return summary


# ============================================================
# 8. SAVE DETAILED CSV RESULTS
# ============================================================

def save_csv_results(results):

    fieldnames = [
        "configuration",
        "question_id",
        "MRR",
        "Strict_MRR",
        "Precision@3",
        "Recall@3",
        "nDCG@3",
        "AvgRelevance@3",
        "HitRate@3",
        "Precision@5",
        "Recall@5",
        "nDCG@5",
        "AvgRelevance@5",
        "HitRate@5",
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

    print(
        f"\nDetailed results saved: {RESULTS_CSV}"
    )


# ============================================================
# 9. SAVE CONFIGURATION SUMMARY
# ============================================================

def save_summary(summary):

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(
        f"Summary saved: {SUMMARY_JSON}"
    )


# ============================================================
# 10. PRINT RESULTS
# ============================================================

def print_results(results, summary):

    print("\n" + "=" * 100)
    print("PASSAGE-LEVEL RETRIEVAL EVALUATION")
    print("=" * 100)

    for config, label in CONFIGURATIONS.items():

        print(f"\n{label.upper()}")
        print("-" * 100)

        config_results = [
            result
            for result in results
            if result["configuration"] == config
        ]

        for result in config_results:

            metrics = result["metrics"]

            print(
                f"\nQuestion: {result['question_id']}"
            )

            print(
                f"MRR: {metrics['MRR']:.4f} | "
                f"Strict MRR: {metrics['Strict_MRR']:.4f}"
            )

            print(
                f"P@3: {metrics['Precision@3']:.4f} | "
                f"R@3: {metrics['Recall@3']:.4f} | "
                f"nDCG@3: {metrics['nDCG@3']:.4f} | "
                f"Hit@3: {metrics['HitRate@3']:.4f}"
            )

            print(
                f"P@5: {metrics['Precision@5']:.4f} | "
                f"R@5: {metrics['Recall@5']:.4f} | "
                f"nDCG@5: {metrics['nDCG@5']:.4f} | "
                f"Hit@5: {metrics['HitRate@5']:.4f} | "
                f"AvgRel@5: {metrics['AvgRelevance@5']:.4f}"
            )

        print(
            f"\n{label.upper()} AVERAGE"
        )

        for metric, value in summary[config].items():

            print(
                f"{metric:<20}: {value:.4f}"
            )


# ============================================================
# 11. MAIN
# ============================================================

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

    # Relevance-grade distribution
    grade_distribution = Counter(
        int(row["relevance_grade"])
        for row in annotations
    )

    print("\n" + "=" * 70)
    print("ANNOTATION DISTRIBUTION")
    print("=" * 70)

    for grade in [0, 1, 2]:

        print(
            f"Grade {grade}: {grade_distribution[grade]}"
        )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
