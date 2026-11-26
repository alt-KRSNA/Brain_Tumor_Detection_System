import csv
from pathlib import Path

from test.evaluate_passage_relevance import (
    evaluate_question,
    calculate_configuration_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASELINE_ANNOTATION_FILE = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "retrieval_baseline"
    / "baseline_relevance_annotations.csv"
)

HYBRID_ANNOTATION_FILE = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "hybrid_retrieval"
    / "hybrid_relevance_annotations.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "hybrid_retrieval"
)

COMPARISON_CSV = REPORT_DIR / "dense_vs_hybrid_comparison.csv"

COMPARISON_MD = REPORT_DIR / "dense_vs_hybrid_comparison.md"

QUESTION_IDS = ["Q01", "Q02", "Q03", "Q04", "Q05"]

RANKS = [1, 2, 3, 4, 5]

METRICS = [
    "Strict_MRR",
    "nDCG@3",
    "nDCG@5",
    "Precision@3",
    "Precision@5",
    "HitRate@3",
    "HitRate@5",
]


def load_grades(file_path, configuration_filter=None):
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    grades = {}

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if (
                configuration_filter is not None
                and row["configuration"] != configuration_filter
            ):
                continue

            question_id = row["question_id"].strip()

            rank = int(row["rank"])

            grade = int(row["relevance_grade"])

            key = (question_id, rank)

            if key in grades:
                raise ValueError(
                    f"Duplicate annotation: {key}"
                )

            grades[key] = grade

    expected = {
        (question_id, rank)
        for question_id in QUESTION_IDS
        for rank in RANKS
    }

    if set(grades) != expected:
        missing = expected - set(grades)
        extra = set(grades) - expected

        raise ValueError(
            f"Annotation mismatch.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    return grades


def evaluate_method(grades_by_question):
    question_results = []

    for question_id in QUESTION_IDS:
        grades = [
            grades_by_question[(question_id, rank)]
            for rank in RANKS
        ]

        metrics = evaluate_question(grades)

        question_results.append({
            "question_id": question_id,
            "grades": grades,
            "metrics": metrics,
        })

    summary = calculate_configuration_summary(
        question_results
    )

    return question_results, summary


def calculate_comparison(dense_results, hybrid_results):
    comparison = []

    for dense, hybrid in zip(
        dense_results,
        hybrid_results,
    ):
        row = {
            "question_id": dense["question_id"],
        }

        for metric in METRICS:
            dense_value = dense["metrics"][metric]

            hybrid_value = hybrid["metrics"][metric]

            row[f"Dense_{metric}"] = dense_value

            row[f"Hybrid_{metric}"] = hybrid_value

            row[f"Delta_{metric}"] = (
                hybrid_value - dense_value
            )

        comparison.append(row)

    return comparison


def save_csv(comparison):
    fieldnames = ["question_id"]

    for metric in METRICS:
        fieldnames.extend([
            f"Dense_{metric}",
            f"Hybrid_{metric}",
            f"Delta_{metric}",
        ])

    with COMPARISON_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(comparison)

    print(f"CSV saved: {COMPARISON_CSV}")


def save_markdown(
    comparison,
    dense_summary,
    hybrid_summary,
):
    lines = [
        "# Dense vs Hybrid Retrieval Comparison",
        "",
        "Dense uses configuration A (1000/200 chunking).",
        "Hybrid uses the same collection and chunking.",
        "",
        "## Aggregate Comparison",
        "",
        "| Metric | Dense | Hybrid | Delta |",
        "|---|---:|---:|---:|",
    ]

    for metric in METRICS:
        dense_value = dense_summary[metric]

        hybrid_value = hybrid_summary[metric]

        delta = hybrid_value - dense_value

        lines.append(
            f"| {metric} "
            f"| {dense_value:.4f} "
            f"| {hybrid_value:.4f} "
            f"| {delta:+.4f} |"
        )

    lines.extend([
        "",
        "## Per-Question Comparison",
        "",
        "| Question | Metric | Dense | Hybrid | Delta |",
        "|---|---|---:|---:|---:|",
    ])

    for row in comparison:
        for metric in METRICS:
            lines.append(
                f"| {row['question_id']} "
                f"| {metric} "
                f"| {row[f'Dense_{metric}']:.4f} "
                f"| {row[f'Hybrid_{metric}']:.4f} "
                f"| {row[f'Delta_{metric}']:+.4f} |"
            )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Positive delta means Hybrid improved.",
        "- Negative delta means Hybrid regressed.",
        "- Zero means no change.",
        "- Only five benchmark questions were evaluated.",
        "- Results are preliminary and not statistically conclusive.",
        "- Human relevance labels are based on the same rubric.",
        "",
    ])

    COMPARISON_MD.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Markdown saved: {COMPARISON_MD}")


def print_comparison(
    comparison,
    dense_summary,
    hybrid_summary,
):
    print("\n" + "=" * 100)
    print("DENSE VS HYBRID — AGGREGATE")
    print("=" * 100)

    print(
        f"{'Metric':<18}"
        f"{'Dense':>12}"
        f"{'Hybrid':>12}"
        f"{'Delta':>12}"
    )

    print("-" * 54)

    for metric in METRICS:
        dense_value = dense_summary[metric]

        hybrid_value = hybrid_summary[metric]

        delta = hybrid_value - dense_value

        print(
            f"{metric:<18}"
            f"{dense_value:>12.4f}"
            f"{hybrid_value:>12.4f}"
            f"{delta:>+12.4f}"
        )

    print("\n" + "=" * 100)
    print("PER-QUESTION STRICT MRR AND nDCG@5")
    print("=" * 100)

    for row in comparison:
        question_id = row["question_id"]

        print(f"\n{question_id}")

        for metric in ["Strict_MRR", "nDCG@5"]:
            delta = row[f"Delta_{metric}"]

            if delta > 0:
                status = "IMPROVED"
            elif delta < 0:
                status = "REGRESSED"
            else:
                status = "UNCHANGED"

            print(
                f"{metric}: "
                f"Dense={row[f'Dense_{metric}']:.4f} | "
                f"Hybrid={row[f'Hybrid_{metric}']:.4f} | "
                f"Delta={delta:+.4f} | "
                f"{status}"
            )


def main():
    print("Loading Dense baseline annotations...")

    dense_grades = load_grades(
        BASELINE_ANNOTATION_FILE,
        configuration_filter="A - Baseline",
    )

    print("Loading Hybrid annotations...")

    hybrid_grades = load_grades(
        HYBRID_ANNOTATION_FILE,
        configuration_filter="Hybrid",
    )

    dense_results, dense_summary = evaluate_method(
        dense_grades
    )

    hybrid_results, hybrid_summary = evaluate_method(
        hybrid_grades
    )

    comparison = calculate_comparison(
        dense_results,
        hybrid_results,
    )

    print_comparison(
        comparison,
        dense_summary,
        hybrid_summary,
    )

    save_csv(comparison)

    save_markdown(
        comparison,
        dense_summary,
        hybrid_summary,
    )

    print("\nComparison completed.")


if __name__ == "__main__":
    main()