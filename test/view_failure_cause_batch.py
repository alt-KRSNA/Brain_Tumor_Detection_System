
import sys

from test.retrieval_evidence import load_retrieval_evidence


BATCH_SIZE = 10


def get_value(record, *keys, default=None):
    for key in keys:
        if key in record:
            return record[key]
    return default


def main():
    batch_number = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    if batch_number < 1:
        raise ValueError("Batch number must be >= 1")

    records = load_retrieval_evidence()

    review_records = [
        record
        for record in records
        if int(get_value(record, "relevance_grade", "grade", default=-1))
        in (0, 1)
    ]

    start = (batch_number - 1) * BATCH_SIZE
    batch = review_records[start:start + BATCH_SIZE]

    if not batch:
        print(f"No records found for batch {batch_number}")
        return

    print("=" * 100)
    print(f"FAILURE CAUSE REVIEW — BATCH {batch_number}")
    print("=" * 100)
    print(f"Total review records : {len(review_records)}")
    print(f"Batch               : {batch_number}")
    print(f"Records displayed    : {len(batch)}")
    print("=" * 100)

    for index, record in enumerate(batch, start=start + 1):
        print(f"\nRECORD #{index}")
        print("-" * 100)

        print("Configuration :", get_value(record, "configuration", "config"))
        print("Question ID   :", get_value(record, "question_id"))
        print("Rank          :", get_value(record, "rank"))
        print("Chunk ID      :", get_value(record, "chunk_id"))
        print("Source        :", get_value(record, "source", "filename"))
        print("Page          :", get_value(record, "page", "page_number"))
        print("Grade         :", get_value(record, "relevance_grade", "grade"))

        print("\nPASSAGE TEXT")
        print("-" * 100)

        passage = get_value(
            record,
            "text",
            "passage_text",
            "content",
            "passage",
            default="[Passage text field not found]",
        )

        print(passage)
        print("-" * 100)


if __name__ == "__main__":
    main()
