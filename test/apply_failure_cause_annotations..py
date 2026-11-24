
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = ROOT / "test" / "reports" / "failure_cause_annotations.csv"
JSON_PATH = ROOT / "test" / "failure_cause_labels.json"

VALID_CAUSES = {
    "wrong_section",
    "incomplete_context",
    "boilerplate_noise",
    "related_not_answer",
    "off_topic",
    "other",
}

VALID_CONFIDENCE = {"high", "medium", "low"}

KEY_FIELDS = (
    "configuration",
    "question_id",
    "rank",
    "chunk_id",
)


def make_key(record):
    return (
        record["configuration"],
        record["question_id"],
        str(record["rank"]),
        record["chunk_id"],
    )


def validate_annotation(annotation):
    cause = annotation.get("failure_cause")
    confidence = annotation.get("review_confidence")
    note = annotation.get("review_note", "").strip()

    if cause not in VALID_CAUSES:
        raise ValueError(f"Invalid failure cause: {cause}")

    if confidence not in VALID_CONFIDENCE:
        raise ValueError(f"Invalid confidence: {confidence}")

    if not note:
        raise ValueError("Review note cannot be blank")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"Annotation file missing: {JSON_PATH}"
        )

    with CSV_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    with JSON_PATH.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    row_lookup = {
        make_key(row): row
        for row in rows
    }

    updated = 0

    for annotation in annotations:
        validate_annotation(annotation)

        key = make_key(annotation)

        if key not in row_lookup:
            raise KeyError(f"Annotation record not found: {key}")

        row = row_lookup[key]

        # Prevent accidental overwriting of existing annotations.
        if row["failure_cause"].strip():
            raise ValueError(
                f"Record already annotated: {key}"
            )

        row["failure_cause"] = annotation["failure_cause"]
        row["review_note"] = annotation["review_note"].strip()
        row["review_confidence"] = annotation[
            "review_confidence"
        ]

        updated += 1

    # Write only after all annotations pass validation.
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    remaining = sum(
        not row["failure_cause"].strip()
        for row in rows
    )

    print("=" * 60)
    print("FAILURE CAUSE ANNOTATIONS APPLIED")
    print("=" * 60)
    print(f"Updated records : {updated}")
    print(f"Total records   : {len(rows)}")
    print(f"Remaining      : {remaining}")
    print(f"Output         : {CSV_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
