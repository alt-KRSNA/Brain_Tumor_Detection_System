import csv
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = PROJECT_ROOT / "test" / "passage_annotation_sheet.csv"


# ==========================================================
# PROVISIONAL RELEVANCE GRADES
# ==========================================================
# 0 = Irrelevant
# 1 = Partially relevant
# 2 = Directly relevant
#
# These are preliminary model-assisted judgments.
# Review before treating them as ground truth.
# ==========================================================

PROVISIONAL_GRADES = {
    # Configuration A
    ("A", "mri"): [2, 1, 2, 1, 0],
    ("A", "symptoms"): [2, 1, 1, 1, 0],

    # Configuration B
    ("B", "glioma"): [1, 2, 1, 1, 0],
    ("B", "meningioma"): [1, 2, 2, 1, 1],
    ("B", "pituitary"): [1, 1, 2, 1, 1],
    ("B", "mri"): [2, 2, 2, 1, 1],
    ("B", "symptoms"): [2, 2, 1, 0, 1],

    # Configuration C
    ("C", "glioma"): [1, 1, 1, 2, 0],
    ("C", "meningioma"): [1, 1, 2, 1, 2],
    ("C", "pituitary"): [1, 1, 1, 1, 2],
    ("C", "mri"): [1, 2, 2, 2, 2],
    ("C", "symptoms"): [2, 0, 1, 1, 0],
}


def normalize(value):
    return str(value).strip().lower()


def find_column(headers, possible_names):
    normalized_headers = {
        normalize(header): header
        for header in headers
    }

    for name in possible_names:
        if name in normalized_headers:
            return normalized_headers[name]

    return None


def detect_question_category(question):
    question = normalize(question)

    if "glioma" in question:
        return "glioma"

    if "meningioma" in question:
        return "meningioma"

    if "pituitary" in question:
        return "pituitary"

    if "symptom" in question:
        return "symptoms"

    if "mri" in question or "imaging" in question:
        return "mri"

    return None


def detect_configuration(value):
    value = normalize(value)

    if value in {"a", "config a", "configuration a"}:
        return "A"

    if value in {"b", "config b", "configuration b"}:
        return "B"

    if value in {"c", "config c", "configuration c"}:
        return "C"

    # Support values like "A (1000/200)"
    if value.startswith("a "):
        return "A"

    if value.startswith("b "):
        return "B"

    if value.startswith("c "):
        return "C"

    return None


def main():

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Annotation CSV not found: {CSV_FILE}"
        )

    # ------------------------------------------------------
    # Create a safety backup before making modifications
    # ------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_file = CSV_FILE.with_name(
        f"passage_annotation_sheet_backup_{timestamp}.csv"
    )

    shutil.copy2(CSV_FILE, backup_file)

    print(f"Backup created: {backup_file.name}")

    # ------------------------------------------------------
    # Read annotation CSV
    # ------------------------------------------------------

    with CSV_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        headers = reader.fieldnames

        if not headers:
            raise ValueError("CSV headers are missing.")

        rows = list(reader)

    # ------------------------------------------------------
    # Detect relevant columns
    # ------------------------------------------------------

    config_column = find_column(
        headers,
        ["configuration", "config", "chunk_config"],
    )

    question_column = find_column(
        headers,
        ["question", "query", "question_id", "query_id"],
    )

    rank_column = find_column(
        headers,
        ["rank", "retrieval_rank", "position"],
    )

    grade_column = find_column(
        headers,
        ["relevance_grade", "grade", "relevance"],
    )

    notes_column = find_column(
        headers,
        ["annotation_notes", "notes", "comments"],
    )

    required_columns = {
        "configuration": config_column,
        "question": question_column,
        "rank": rank_column,
        "grade": grade_column,
    }

    missing = [
        name
        for name, column in required_columns.items()
        if column is None
    ]

    if missing:
        raise ValueError(
            f"Could not identify CSV columns: {missing}\n"
            f"Available columns: {headers}"
        )

    # ------------------------------------------------------
    # Fill only blank grades
    # ------------------------------------------------------

    filled = 0
    skipped_existing = 0
    skipped_unknown = 0

    for row in rows:

        existing_grade = normalize(
            row.get(grade_column, "")
        )

        # Never overwrite existing annotations
        if existing_grade:
            skipped_existing += 1
            continue

        config = detect_configuration(
            row.get(config_column, "")
        )

        category = detect_question_category(
            row.get(question_column, "")
        )

        try:
            rank = int(
                row.get(rank_column, "")
            )
        except (ValueError, TypeError):
            skipped_unknown += 1
            continue

        key = (config, category)

        if key not in PROVISIONAL_GRADES:
            skipped_unknown += 1
            continue

        grades = PROVISIONAL_GRADES[key]

        if not 1 <= rank <= len(grades):
            skipped_unknown += 1
            continue

        grade = grades[rank - 1]

        row[grade_column] = str(grade)

        if notes_column:
            row[notes_column] = (
                "Provisional model-assisted grade; "
                "manual review required."
            )

        filled += 1

    # ------------------------------------------------------
    # Write updated CSV
    # ------------------------------------------------------

    with CSV_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=headers,
        )

        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    remaining = sum(
        1
        for row in rows
        if not normalize(row.get(grade_column, ""))
    )

    completed = len(rows) - remaining

    print("\n" + "=" * 60)
    print("PROVISIONAL ANNOTATION FILL COMPLETED")
    print("=" * 60)

    print(f"Total rows              : {len(rows)}")
    print(f"New provisional grades  : {filled}")
    print(f"Existing grades preserved: {skipped_existing}")
    print(f"Skipped/unknown rows    : {skipped_unknown}")
    print(f"Completed annotations   : {completed}/{len(rows)}")
    print(f"Remaining blank grades  : {remaining}")

    print(f"\nUpdated CSV: {CSV_FILE}")

    print("\nIMPORTANT:")
    print("These grades are provisional, not ground truth.")
    print("Review the CSV before importing into JSON.")
    print("No JSON annotations were modified by this script.")


if __name__ == "__main__":
    main()