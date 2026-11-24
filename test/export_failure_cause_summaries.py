
import csv
from pathlib import Path
from collections import Counter, defaultdict


def export_failure_cause_summaries(csv_path, output_dir):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Annotation CSV not found: {csv_path}"
        )

    with csv_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    category_counts = Counter()
    configuration_counts = defaultdict(Counter)
    question_counts = defaultdict(Counter)
    confidence_counts = Counter()

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

    output_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(filename, fieldnames, data):
        path = output_dir / filename

        with path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(data)

        return path

    # 1. Overall category summary
    overall_data = []

    for cause, count in category_counts.most_common():
        percentage = (
            count / reviewed * 100 if reviewed else 0
        )

        overall_data.append({
            "failure_cause": cause,
            "count": count,
            "percentage": round(percentage, 2),
        })

    overall_path = write_csv(
        "failure_cause_summary.csv",
        ["failure_cause", "count", "percentage"],
        overall_data,
    )

    # 2. Configuration-wise summary
    configuration_data = []

    for config in sorted(configuration_counts):
        total = sum(configuration_counts[config].values())

        for cause, count in sorted(
            configuration_counts[config].items()
        ):
            percentage = count / total * 100 if total else 0

            configuration_data.append({
                "configuration": config,
                "failure_cause": cause,
                "count": count,
                "percentage": round(percentage, 2),
            })

    configuration_path = write_csv(
        "failure_cause_by_configuration.csv",
        [
            "configuration",
            "failure_cause",
            "count",
            "percentage",
        ],
        configuration_data,
    )

    # 3. Question-wise summary
    question_data = []

    for question in sorted(question_counts):
        total = sum(question_counts[question].values())

        for cause, count in sorted(
            question_counts[question].items()
        ):
            percentage = count / total * 100 if total else 0

            question_data.append({
                "question_id": question,
                "failure_cause": cause,
                "count": count,
                "percentage": round(percentage, 2),
            })

    question_path = write_csv(
        "failure_cause_by_question.csv",
        [
            "question_id",
            "failure_cause",
            "count",
            "percentage",
        ],
        question_data,
    )

    # 4. Confidence summary
    confidence_data = [
        {
            "confidence": confidence,
            "count": count,
        }
        for confidence, count in sorted(
            confidence_counts.items()
        )
    ]

    confidence_path = write_csv(
        "failure_cause_by_confidence.csv",
        ["confidence", "count"],
        confidence_data,
    )

    return {
        "overall": overall_path,
        "configuration": configuration_path,
        "question": question_path,
        "confidence": confidence_path,
    }

