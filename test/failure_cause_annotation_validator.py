# test/failure_cause_annotation_validator.py

import csv
from pathlib import Path

ALLOWED_CAUSES = {
    "wrong_section",
    "incomplete_context",
    "boilerplate_noise",
    "related_not_answer",
    "off_topic",
    "other",
}

ALLOWED_CONFIDENCE = {"high", "medium", "low"}

REQUIRED_COLUMNS = {
    "configuration",
    "question_id",
    "rank",
    "chunk_id",
    "relevance_grade",
    "failure_cause",
    "review_note",
    "review_confidence",
}


def validate_failure_cause_annotations(
    csv_path,
    expected_total=50,
    expected_unreviewed=1,
):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    errors = []

    missing_columns = REQUIRED_COLUMNS - columns
    if missing_columns:
        return {
            "valid": False,
            "total_records": len(rows),
            "reviewed_records": 0,
            "unreviewed_records": 0,
            "errors": [
                f"Missing columns: {sorted(missing_columns)}"
            ],
        }

    if len(rows) != expected_total:
        errors.append(
            f"Expected {expected_total} records, found {len(rows)}."
        )

    seen = set()
    reviewed = 0
    unreviewed = 0

    for index, row in enumerate(rows, start=2):
        key = (
            row["configuration"].strip(),
            row["question_id"].strip(),
            row["rank"].strip(),
            row["chunk_id"].strip(),
        )

        if key in seen:
            errors.append(f"Duplicate passage identifier at CSV line {index}: {key}")
        seen.add(key)

        if row["relevance_grade"].strip() not in {"0", "1"}:
            errors.append(
                f"Invalid relevance grade at CSV line {index}: "
                f"{row['relevance_grade']!r}"
            )

        cause = row["failure_cause"].strip()
        note = row["review_note"].strip()
        confidence = row["review_confidence"].strip()

        fields = (cause, note, confidence)

        if all(value == "" for value in fields):
            unreviewed += 1
            continue

        reviewed += 1

        if not cause:
            errors.append(f"Missing failure_cause at CSV line {index}.")
        elif cause not in ALLOWED_CAUSES:
            errors.append(
                f"Invalid failure_cause at CSV line {index}: {cause}"
            )

        if not note:
            errors.append(f"Missing review_note at CSV line {index}.")

        if not confidence:
            errors.append(f"Missing review_confidence at CSV line {index}.")
        elif confidence not in ALLOWED_CONFIDENCE:
            errors.append(
                f"Invalid review_confidence at CSV line {index}: {confidence}"
            )

    if unreviewed != expected_unreviewed:
        errors.append(
            f"Expected {expected_unreviewed} unreviewed records, "
            f"found {unreviewed}."
        )

    return {
        "valid": len(errors) == 0,
        "total_records": len(rows),
        "reviewed_records": reviewed,
        "unreviewed_records": unreviewed,
        "errors": errors,
    }