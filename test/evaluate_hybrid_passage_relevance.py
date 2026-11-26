import csv
import json
from collections import Counter
from pathlib import Path

from test.evaluate_passage_relevance import (
    evaluate_question,
    calculate_configuration_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "hybrid_retrieval"
)

ANNOTATION_FILE = (
    REPORT_DIR
    / "hybrid_relevance_annotations.csv"
)

RESULTS_CSV = (
    REPORT_DIR
    / "hybrid_passage_evaluation.csv"
)

SUMMARY_JSON = (
    REPORT_DIR
    / "hybrid_passage_summary.json"
)

SUMMARY_MD = (
    REPORT_DIR
    / "hybrid_passage_summary.md"
)

QUESTION_IDS = ["Q01", "Q02", "Q03", "Q04", "Q05"]

EXPECTED_RANKS = [1, 2, 3, 4, 5]

REQUIRED_COLUMNS = {
    "configuration",
    "question_id",
    "rank",
    "passage_id",
    "source",
    "page",
    "passage_text",
    "relevance_grade",
    "review_note",
    "review_confidence",
}


def load_and_validate_annotations():
    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_FILE}"
        )

    with ANNOTATION_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        columns = set(reader.fieldnames or [])

        if not REQUIRED_COLUMNS.issubset(columns):
            missing = REQUIRED_COLUMNS - columns

            raise ValueError(
                f"Missing columns: {missing}"
            )

        annotations = list(reader)

    if len(annotations) != 25:
        raise ValueError(
            f"Expected 25 annotations, got {len(annotations)}."
        )

    seen = set()

    for row in annotations:
        question_id = row["question_id"].strip()

        rank = int(row["rank"])

        grade = int(row["relevance_grade"])

        key = (question_id, rank)

        if question_id not in QUESTION_IDS:
            raise ValueError(
                f"Unknown question ID: {question_id}"
            )

        if rank not in EXPECTED_RANKS:
            raise ValueError(
                f"Invalid rank: {rank}"
            )

        if grade not in (0, 1, 2):
            raise ValueError(
                f"Invalid relevance grade: {grade}"
            )

        if key in seen:
            raise ValueError(
                f"Duplicate annotation: {key}"
            )

        seen.add(key)

        if not row["passage_id"].strip():
            raise ValueError(
                f"Missing passage ID: {key}"
            )

        if not row["passage_text"].strip():
            raise ValueError(
                f"Missing passage text: {key}"
            )

        if not row["review_note"].strip():
            raise ValueError(
                f"Missing review note: {key}"
            )

        if row["review_confidence"].strip().lower() not in {
            "high", "medium", "low"
        }:
            raise ValueError(
                f"Invalid confidence: {key}"
            )

    expected = {
        (question_id, rank)
        for question_id in QUESTION_IDS
        for rank in EXPECTED_RANKS
    }

    if seen != expected:
        missing = expected - seen

        raise ValueError(
            f"Missing annotations: {sorted(missing)}"
        )

    print("Validation passed: all 25 Hybrid passages are graded.")

    return annotations


def organize_annotations(annotations):
    organized = {}

    for row in annotations:
        question_id = row["question_id"]

        rank = int(row["rank"])

        grade = int(row["relevance_grade"])

        organized.setdefault(question_id, {})[rank] = grade

    return organized


def calculate_results(organized):
    results = []

    question_results = []

    for question_id in QUESTION_IDS:
        grades = [
            organized[question_id][rank]
            for rank in EXPECTED_RANKS
        ]

        metrics = evaluate_question(grades)

        result = {
            "configuration": "Hybrid",
            "question_id": question_id,
            "grades": grades,
            "metrics": metrics,
        }

        results.append(result)
        question_results.append(result)

    summary = calculate_configuration_summary(
        question_results
    )

    return results, summary


def save_results(results):
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
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            row = {
                "configuration": "Hybrid",
                "question_id": result["question_id"],
            }

            row.update(result["metrics"])

            writer.writerow(row)

    print(f"CSV saved: {RESULTS_CSV}")


def save_summary(summary):
    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print(f"JSON saved: {SUMMARY_JSON}")


def save_markdown(results, summary, annotations):
    grade_distribution = Counter(
        int(row["relevance_grade"])
        for row in annotations
    )

    lines = [
        "# Hybrid Passage-Level Evaluation",
        "",
        "## Aggregate Results",
        "",
        "| Metric | Mean |",
        "|---|---:|",
    ]

    for metric, value in summary.items():
        lines.append(
            f"| {metric} | {value:.4f} |"
        )

    lines.extend([
        "",
        "## Per-Question Results",
        "",
        "| Question | Strict MRR | nDCG@3 | nDCG@5 | P@3 | P@5 | Hit@3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])

    for result in results:
        metrics = result["metrics"]

        lines.append(
            f"| {result['question_id']} "
            f"| {metrics['Strict_MRR']:.4f} "
            f"| {metrics['nDCG@3']:.4f} "
            f"| {metrics['nDCG@5']:.4f} "
            f"| {metrics['Precision@3']:.4f} "
            f"| {metrics['Precision@5']:.4f} "
            f"| {metrics['HitRate@3']:.4f} |"
        )

    lines.extend([
        "",
        "## Annotation Distribution",
        "",
        f"- Grade 0: {grade_distribution[0]}",
        f"- Grade 1: {grade_distribution[1]}",
        f"- Grade 2: {grade_distribution[2]}",
        "",
        "## Methodology",
        "",
        "- Five benchmark questions.",
        "- Five Hybrid retrieved passages per question.",
        "- 25 human relevance judgments.",
        "- Grade 0 = irrelevant.",
        "- Grade 1 = related but insufficient.",
        "- Grade 2 = directly useful.",
        "- Precision counts grades 1 and 2 as relevant.",
        "- Strict MRR and HitRate count only grade 2.",
        "",
        "This is a small preliminary evaluation.",
        "It should not be treated as statistically conclusive.",
    ])

    SUMMARY_MD.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Markdown saved: {SUMMARY_MD}")


def print_results(results, summary, annotations):
    print("\n" + "=" * 90)
    print("HYBRID PASSAGE-LEVEL EVALUATION")
    print("=" * 90)

    for result in results:
        metrics = result["metrics"]

        print(f"\n{result['question_id']}")

        print(
            f"Strict MRR: {metrics['Strict_MRR']:.4f} | "
            f"nDCG@3: {metrics['nDCG@3']:.4f} | "
            f"nDCG@5: {metrics['nDCG@5']:.4f}"
        )

        print(
            f"P@3: {metrics['Precision@3']:.4f} | "
            f"P@5: {metrics['Precision@5']:.4f} | "
            f"Hit@3: {metrics['HitRate@3']:.4f}"
        )

    print("\n" + "=" * 90)
    print("HYBRID AGGREGATE")
    print("=" * 90)

    for metric, value in summary.items():
        print(f"{metric:<20}: {value:.4f}")

    distribution = Counter(
        int(row["relevance_grade"])
        for row in annotations
    )

    print("\nAnnotation Distribution:")

    for grade in [0, 1, 2]:
        print(
            f"Grade {grade}: {distribution[grade]}"
        )


def main():
    annotations = load_and_validate_annotations()

    organized = organize_annotations(annotations)

    results, summary = calculate_results(organized)

    print_results(results, summary, annotations)

    save_results(results)

    save_summary(summary)

    save_markdown(results, summary, annotations)

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()