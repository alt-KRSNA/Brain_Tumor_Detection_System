
import csv
from pathlib import Path
from collections import Counter, defaultdict


def analyze_failure_causes(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")

    with csv_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    category_counts = Counter()
    confidence_counts = Counter()

    configuration_counts = defaultdict(Counter)
    question_counts = defaultdict(Counter)

    reviewed_records = 0
    unreviewed_records = 0

    for row in rows:
        cause = row["failure_cause"].strip()
        confidence = row["review_confidence"].strip()
        configuration = row["configuration"].strip()
        question_id = row["question_id"].strip()

        if not cause and not confidence:
            unreviewed_records += 1
            continue

        reviewed_records += 1

        if cause:
            category_counts[cause] += 1
            configuration_counts[configuration][cause] += 1
            question_counts[question_id][cause] += 1

        if confidence:
            confidence_counts[confidence] += 1

    return {
        "total_records": len(rows),
        "reviewed_records": reviewed_records,
        "unreviewed_records": unreviewed_records,
        "category_counts": dict(category_counts),
        "configuration_counts": {
            config: dict(counts)
            for config, counts in configuration_counts.items()
        },
        "question_counts": {
            question: dict(counts)
            for question, counts in question_counts.items()
        },
        "confidence_counts": dict(confidence_counts),
    }
