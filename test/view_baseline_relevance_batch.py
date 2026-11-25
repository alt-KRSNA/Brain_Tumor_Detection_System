from pathlib import Path
import csv
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANNOTATION_CSV = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "retrieval_baseline"
    / "baseline_relevance_annotations.csv"
)

BATCH_SIZE = 5


def load_annotation_records(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Annotation CSV not found: {csv_path}"
        )

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    required_fields = {
        "configuration",
        "question_id",
        "question",
        "rank",
        "chunk_id",
        "source",
        "page",
        "distance",
        "text",
        "relevance_grade",
        "review_note",
        "review_confidence",
    }

    missing = required_fields - set(
        reader.fieldnames or []
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # Group passages by configuration and question.
    groups = {}

    for row in rows:
        key = (
            row["configuration"],
            row["question_id"],
        )

        groups.setdefault(key, []).append(row)

    # Sort passages within each question by retrieval rank.
    for group_rows in groups.values():
        group_rows.sort(
            key=lambda row: int(row["rank"])
        )

    # Keep the question groups in their original CSV order.
    return list(groups.values())


def get_batch(records, batch_number):
    if batch_number < 1:
        raise ValueError(
            "Batch number must start from 1."
        )

    total_batches = len(records)

    if batch_number > total_batches:
        raise ValueError(
            f"Batch {batch_number} does not exist. "
            f"Total batches: {total_batches}"
        )

    return records[batch_number - 1]


def display_batch(group_rows, batch_number):
    if not group_rows:
        raise ValueError(
            "Batch contains no records."
        )

    configuration = group_rows[0]["configuration"]
    question_id = group_rows[0]["question_id"]
    question = group_rows[0]["question"]

    reviewed = sum(
        row["relevance_grade"].strip() != ""
        for row in group_rows
    )

    print("=" * 100)
    print("BASELINE RELEVANCE REVIEW")
    print("=" * 100)

    print(f"Batch         : {batch_number}")
    print(f"Configuration : {configuration}")
    print(f"Question ID   : {question_id}")
    print(f"Question      : {question}")
    print(f"Passages      : {len(group_rows)}")
    print(f"Reviewed      : {reviewed}/{len(group_rows)}")

    print("\nRelevance grades:")
    print("0 = Irrelevant")
    print("1 = Related but insufficient to answer")
    print("2 = Directly useful for answering")

    print("\nConfidence:")
    print("high / medium / low")

    print("=" * 100)

    for row in group_rows:
        print(
            f"\nPASSAGE RANK #{row['rank']}"
        )

        print("-" * 100)

        print(f"Chunk ID : {row['chunk_id']}")
        print(f"Source   : {row['source']}")
        print(f"Page     : {row['page']}")
        print(f"Distance : {row['distance']}")

        print("\nPassage text:")
        print(row["text"])

        print("\nCurrent annotation:")
        print(
            f"Grade      : {row['relevance_grade'] or 'UNREVIEWED'}"
        )
        print(
            f"Confidence  : {row['review_confidence'] or 'UNREVIEWED'}"
        )
        print(
            f"Review note: {row['review_note'] or 'NONE'}"
        )

        print("-" * 100)

    print("\nEND OF BATCH")
    print("=" * 100)


def main():
    batch_number = (
        int(sys.argv[1])
        if len(sys.argv) > 1
        else 1
    )

    groups = load_annotation_records(
        ANNOTATION_CSV
    )

    group_rows = get_batch(
        groups,
        batch_number,
    )

    display_batch(
        group_rows,
        batch_number,
    )


if __name__ == "__main__":
    main()