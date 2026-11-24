import csv
from pathlib import Path


ANNOTATION_COLUMNS = [
    "configuration",
    "question_id",
    "rank",
    "chunk_id",
    "source",
    "page",
    "relevance_grade",
    "failure_cause",
    "review_note",
    "review_confidence",
]


def generate_failure_cause_template(records, output_path):
    """
    Generate a CSV template for manually reviewing retrieval failures.

    Only passages with relevance grades 0 and 1 are included.

    Grade 0: Irrelevant passage
    Grade 1: Partially relevant passage
    Grade 2: Directly relevant passage
    """

    failed_records = [
        record
        for record in records
        if int(record["relevance_grade"]) in (0, 1)
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=ANNOTATION_COLUMNS
        )

        writer.writeheader()

        for record in failed_records:
            row = {
                column: record.get(column, "")
                for column in ANNOTATION_COLUMNS
            }

            # Leave manual annotation fields empty.
            row["failure_cause"] = ""
            row["review_note"] = ""
            row["review_confidence"] = ""

            writer.writerow(row)

    return len(failed_records)