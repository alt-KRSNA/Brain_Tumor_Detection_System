
import csv
from pathlib import Path
from collections import Counter, defaultdict


def generate_failure_cause_report(csv_path, output_path):
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Annotation CSV not found: {csv_path}"
        )

    with csv_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    category_counts = Counter()
    confidence_counts = Counter()
    configuration_counts = defaultdict(Counter)
    question_counts = defaultdict(Counter)

    reviewed = 0
    unreviewed = 0

    for row in rows:
        cause = row["failure_cause"].strip()
        confidence = row["review_confidence"].strip()

        if not cause and not confidence:
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

    lines = [
        "# Retrieval Failure Cause Analysis",
        "",
        "## 1. Dataset Overview",
        "",
        f"- Total records: {len(rows)}",
        f"- Reviewed records: {reviewed}",
        f"- Unreviewed records: {unreviewed}",
        "",
        "## 2. Overall Failure-Cause Distribution",
        "",
        "| Failure Cause | Count | Percentage |",
        "|---|---:|---:|",
    ]

    for cause, count in category_counts.most_common():
        percentage = (
            count / reviewed * 100 if reviewed else 0
        )
        lines.append(
            f"| {cause} | {count} | {percentage:.2f}% |"
        )

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

    lines.extend([
        "## 5. Review Confidence Distribution",
        "",
        "| Confidence | Count |",
        "|---|---:|",
    ])

    for confidence, count in sorted(
        confidence_counts.items()
    ):
        lines.append(f"| {confidence} | {count} |")

    lines.extend([
        "",
        "## 6. Interpretation",
        "",
        "The most frequent annotated failure category is "
        "the dominant observed retrieval issue in this dataset.",
        "",
        "Configuration-wise and question-wise counts are "
        "descriptive. They do not establish causation or prove "
        "that one retrieval configuration is universally better.",
        "",
        "## 7. Limitations",
        "",
        "- The evaluation contains only 50 candidate passages.",
        "- One passage remains unreviewed.",
        "- Some annotations are model-assisted and require "
        "human verification.",
        "- Results depend on the selected questions, documents, "
        "and retrieved passages.",
        "- These findings should not be generalized without "
        "a larger evaluation.",
        "",
    ])

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return output_path
