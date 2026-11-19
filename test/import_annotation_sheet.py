import csv
import json
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = PROJECT_ROOT / "test" / "passage_annotation_sheet.csv"
JSON_FILE = PROJECT_ROOT / "test" / "passage_annotations.json"

CONFIGURATIONS = ["A", "B", "C"]
QUESTION_IDS = ["glioma", "meningioma", "pituitary", "mri", "symptoms"]
EXPECTED_TOTAL = 75


def parse_grade(value):
    """Convert a CSV grade to an integer or None."""

    value = value.strip()

    if not value:
        return None

    try:
        grade = int(value)
    except ValueError:
        raise ValueError(f"Invalid grade value: {value!r}")

    if grade not in (0, 1, 2):
        raise ValueError(f"Grade must be 0, 1, or 2. Got: {grade}")

    return grade


def main():

    if not CSV_FILE.exists():
        raise FileNotFoundError(CSV_FILE)

    if not JSON_FILE.exists():
        raise FileNotFoundError(JSON_FILE)

    # Load existing JSON annotations
    with JSON_FILE.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    # Create a backup before modifying anything
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_file = JSON_FILE.with_name(
        f"passage_annotations_backup_{timestamp}.json"
    )

    shutil.copy2(JSON_FILE, backup_file)

    print(f"Backup created: {backup_file.name}")

    imported = 0
    skipped_blank = 0

    # Read annotation sheet
    with CSV_FILE.open(
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
        }

        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                f"Missing CSV columns: {missing_columns}"
            )

        for row in reader:

            config = row["configuration"].strip()
            question_id = row["question_id"].strip()
            rank = int(row["rank"])
            grade = parse_grade(row["relevance_grade"])

            if grade is None:
                skipped_blank += 1
                continue

            if config not in CONFIGURATIONS:
                raise ValueError(f"Unknown configuration: {config}")

            if question_id not in QUESTION_IDS:
                raise ValueError(f"Unknown question: {question_id}")

            key = f"{config}|{question_id}|{rank}"

            # Preserve existing annotations unless CSV has a grade.
            # CSV grades take precedence for entries being imported.
            annotations[key] = {
                "configuration": config,
                "question_id": question_id,
                "rank": rank,
                "relevance_grade": grade,
            }

            imported += 1

    # Save merged annotations
    with JSON_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            annotations,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Count completed annotations
    completed = 0

    for config in CONFIGURATIONS:
        for question_id in QUESTION_IDS:
            for rank in range(1, 6):

                key = f"{config}|{question_id}|{rank}"

                entry = annotations.get(key)

                if (
                    isinstance(entry, dict)
                    and entry.get("relevance_grade") in (0, 1, 2)
                ):
                    completed += 1

    remaining = EXPECTED_TOTAL - completed

    print("\n" + "=" * 65)
    print("ANNOTATION IMPORT COMPLETED")
    print("=" * 65)

    print(f"Grades imported from CSV : {imported}")
    print(f"Blank rows skipped       : {skipped_blank}")
    print(f"Total completed          : {completed}/{EXPECTED_TOTAL}")
    print(f"Remaining annotations    : {remaining}")
    print(f"Updated JSON             : {JSON_FILE}")

    if remaining == 0:
        print("\nAll annotations are ready for evaluation!")
    else:
        print("\nComplete the remaining grades in the CSV sheet.")
        print("Then rerun this importer.")


if __name__ == "__main__":
    main()