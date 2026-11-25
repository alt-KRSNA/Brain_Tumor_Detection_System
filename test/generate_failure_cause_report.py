
import csv
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "test"
    / "reports"
)

INPUT_FILE = (
    REPORT_DIRECTORY
    / "failure_cause_annotations.csv"
)

OUTPUT_FILE = (
    REPORT_DIRECTORY
    / "failure_cause_report.md"
)


# ============================================================
# REPORT GENERATOR
# ============================================================

def generate_failure_cause_report(csv_path, output_path):

    csv_path = Path(csv_path)
    output_path = Path(output_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Annotation CSV not found: {csv_path}"
        )

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        rows = list(csv.DictReader(file))

    required_columns = {
        "failure_cause",
        "review_confidence",
        "configuration",
        "question_id",
    }

    if rows and not required_columns.issubset(rows[0].keys()):
        raise ValueError(
            f"CSV is missing required columns: "
            f"{required_columns - set(rows[0].keys())}"
        )

    category_counts = Counter()
    confidence_counts = Counter()

    configuration_counts = defaultdict(Counter)
    question_counts = defaultdict(Counter)

    reviewed = 0
    unreviewed = 0

    for row in rows:

        cause = row["failure_cause"].strip()
        confidence = row["review_confidence"].strip()
        note = row.get("review_note", "").strip()

        if not cause and not confidence and not note:
            unreviewed += 1
            continue

        reviewed += 1

        if cause:

            category_counts[cause] += 1

            configuration_counts[
                row["configuration"].strip()
            ][cause] += 1

            question_counts[
                row["question_id"].strip()
            ][cause] += 1

        if confidence:
            confidence_counts[confidence] += 1

    total_causes = sum(category_counts.values())

    # --------------------------------------------------------
    # BUILD REPORT
    # --------------------------------------------------------

    lines = [
        "# Retrieval Failure Cause Analysis",
        "",
        "## 1. Dataset Overview",
        "",
        f"- Input file: `{csv_path.name}`",
        f"- Total records: {len(rows)}",
        f"- Reviewed records: {reviewed}",
        f"- Unreviewed records: {unreviewed}",
        f"- Records with failure-cause labels: {total_causes}",
        "",
        "## 2. Overall Failure-Cause Distribution",
        "",
        "| Failure Cause | Count | Percentage |",
        "|---|---:|---:|",
    ]

    for cause, count in category_counts.most_common():

        percentage = (
            count / total_causes * 100
            if total_causes else 0
        )

        lines.append(
            f"| {cause} | {count} | {percentage:.2f}% |"
        )

    # Configuration-wise analysis

    lines.extend([
        "",
        "## 3. Configuration-wise Analysis",
        "",
    ])

    for config in sorted(configuration_counts):

        lines.extend([
            f"### Configuration {config}",
            "",
            "| Failure Cause | Count |",
            "|---|---:|",
        ])

        for cause, count in configuration_counts[
            config
        ].most_common():

            lines.append(f"| {cause} | {count} |")

        lines.append("")

    # Question-wise analysis

    lines.extend([
        "## 4. Question-wise Analysis",
        "",
    ])

    for question in sorted(question_counts):

        lines.extend([
            f"### {question}",
            "",
            "| Failure Cause | Count |",
            "|---|---:|",
        ])

        for cause, count in question_counts[
            question
        ].most_common():

            lines.append(f"| {cause} | {count} |")

        lines.append("")

    # Confidence distribution

    lines.extend([
        "## 5. Review Confidence Distribution",
        "",
        "| Confidence | Count |",
        "|---|---:|",
    ])

    for confidence, count in sorted(
        confidence_counts.items()
    ):

        lines.append(
            f"| {confidence} | {count} |"
        )

    # Interpretation and limitations

    lines.extend([
        "",
        "## 6. Interpretation",
        "",
        "The most frequent annotated failure category represents "
        "the dominant observed retrieval issue in this dataset.",
        "",
        "Configuration-wise and question-wise counts are "
        "descriptive. They do not establish causation or prove "
        "that one retrieval configuration is universally better.",
        "",
        "## 7. Limitations",
        "",
        f"- The analysis uses {len(rows)} annotated candidate passages.",
        "- Findings depend on the selected questions, documents, "
        "retrieved passages, and annotation quality.",
        "- Failure-cause labels describe observed cases and do "
        "not independently establish causation.",
        "- A larger evaluation dataset is needed before "
        "generalizing these findings.",
        "",
    ])

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RETRIEVAL FAILURE CAUSE REPORT GENERATOR")
    print("=" * 70)

    print(f"\nInput CSV: {INPUT_FILE}")

    output_file = generate_failure_cause_report(
        INPUT_FILE,
        OUTPUT_FILE
    )

    print("\nReport generated successfully.")
    print(f"Output: {output_file}")

    print("\nReport statistics:")

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        total_records = sum(1 for _ in csv.DictReader(file))

    print(f"Total input records: {total_records}")


if __name__ == "__main__":
    main()
