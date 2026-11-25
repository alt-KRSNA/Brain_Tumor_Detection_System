
import csv
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "retrieval_baseline"
    / "baseline_relevance_annotations.csv"
)

# Grades are ordered by retrieval rank: 1, 2, 3, 4, 5.
GRADES = {
    ("A - Baseline", "Q01"): [2, 1, 0, 1, 1],
    ("A - Baseline", "Q02"): [1, 1, 1, 1, 2],
    ("A - Baseline", "Q03"): [2, 1, 1, 1, 0],
    ("A - Baseline", "Q04"): [2, 2, 2, 2, 1],
    ("A - Baseline", "Q05"): [2, 2, 1, 1, 0],

    ("B - Medium", "Q01"): [2, 1, 0, 1, 2],
    ("B - Medium", "Q02"): [1, 2, 2, 0, 1],
    ("B - Medium", "Q03"): [1, 1, 2, 1, 1],
    ("B - Medium", "Q04"): [2, 1, 2, 1, 2],
    ("B - Medium", "Q05"): [2, 2, 1, 0, 1],

    ("C - Small", "Q01"): [1, 2, 2, 1, 0],
    ("C - Small", "Q02"): [1, 1, 2, 1, 2],
    ("C - Small", "Q03"): [1, 1, 1, 1, 2],
    ("C - Small", "Q04"): [2, 1, 2, 2, 2],
    ("C - Small", "Q05"): [2, 1, 1, 2, 0],
}


NOTES = {
    0: "Does not provide useful information for answering the question.",
    1: "Related to the question but does not sufficiently answer it.",
    2: "Directly useful information for answering the question.",
}


def main():
    print("=" * 70)
    print("APPLYING HUMAN RELEVANCE ANNOTATIONS")
    print("=" * 70)

    if not CSV_FILE.exists():
        raise FileNotFoundError(CSV_FILE)

    # Read the original CSV.
    with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    required_columns = {
        "configuration",
        "question_id",
        "rank",
        "relevance_grade",
        "review_note",
        "review_confidence",
    }

    if not required_columns.issubset(set(fieldnames or [])):
        raise ValueError("CSV is missing required annotation columns.")

    if len(rows) != 75:
        raise ValueError(
            f"Expected 75 records, found {len(rows)}. Stopping."
        )

    # Never overwrite annotations that may already exist.
    existing = [
        row for row in rows
        if row["relevance_grade"].strip()
        or row["review_note"].strip()
        or row["review_confidence"].strip()
    ]

    if existing:
        raise ValueError(
            f"Found {len(existing)} already annotated records. "
            "Stopping to protect existing work."
        )

    seen = set()

    for row in rows:
        config = row["configuration"].strip()
        question_id = row["question_id"].strip()
        rank = int(row["rank"])

        key = (config, question_id)

        if key not in GRADES:
            raise ValueError(f"Unexpected configuration/question: {key}")

        if rank not in range(1, 6):
            raise ValueError(f"Invalid rank: {rank}")

        record_key = (config, question_id, rank)

        if record_key in seen:
            raise ValueError(f"Duplicate record: {record_key}")

        seen.add(record_key)

        grade = GRADES[key][rank - 1]

        row["relevance_grade"] = str(grade)
        row["review_note"] = NOTES[grade]
        row["review_confidence"] = "high"

    if len(seen) != 75:
        raise ValueError(
            f"Expected 75 unique records, found {len(seen)}."
        )

    # Create a timestamped backup before modifying the original.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = CSV_FILE.with_name(
        f"baseline_relevance_annotations_{timestamp}.backup.csv"
    )

    shutil.copy2(CSV_FILE, backup_file)

    # Write updated annotations while preserving all other columns.
    with CSV_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Verify the saved file.
    with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        saved_rows = list(csv.DictReader(file))

    reviewed = sum(
        bool(row["relevance_grade"].strip())
        for row in saved_rows
    )

    distribution = Counter(
        row["relevance_grade"] for row in saved_rows
    )

    print("\nANNOTATION COMPLETED")
    print("-" * 70)
    print(f"Total records       : {len(saved_rows)}")
    print(f"Reviewed records    : {reviewed}")
    print(f"Missing grades      : {len(saved_rows) - reviewed}")
    print(f"Grade 0             : {distribution['0']}")
    print(f"Grade 1             : {distribution['1']}")
    print(f"Grade 2             : {distribution['2']}")
    print(f"Backup created      : {backup_file.name}")
    print(f"Updated CSV         : {CSV_FILE.name}")

    if reviewed != 75:
        raise ValueError("Verification failed: not all records were saved.")

    print("\nAll 75 annotations verified successfully.")


if __name__ == "__main__":
    main()
