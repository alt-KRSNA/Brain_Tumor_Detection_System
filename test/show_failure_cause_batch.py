
from pathlib import Path
import sys

from test.retrieval_evidence import load_retrieval_evidence


BATCH_SIZE = 10


def main():
    batch_number = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    if batch_number < 1:
        raise ValueError("Batch number must be >= 1")

    records = load_retrieval_evidence()

    # Only Grade 0 and Grade 1 passages require failure-cause analysis.
    review_records = [
        record
        for record in records
        if record["relevance_grade"] in (0, 1)
    ]

    start = (batch_number - 1) * BATCH_SIZE
    end = start + BATCH_SIZE
    batch = review_records[start:end]

    if not batch:
        print(f"No records found for batch {batch_number}")
        return

    print("=" * 100)
    print(f"FAILURE CAUSE REVIEW — BATCH {batch_number}")
    print("=" * 100)
    print(f"Records in dataset : {len(review_records)}")
    print(f"Batch range       : {start + 1}–{min(end, len(review_records))}")
    print(f"Batch records     : {len(batch)}")
    print("=" * 100)

    for index, record in enumerate(batch, start=start + 1):
        print(f"\nRECORD #{index}")
        print("-" * 100)

        print(f"Configuration : {record['configuration']}")
        print(f"Question ID   : {record['question_id']}")
        print(f"Rank          : {record['rank']}")
        print(f"Chunk ID      : {record['chunk_id']}")
        print(f"Source        : {record['source']}")
        print(f"Page          : {record['page']}")
        print(f"Grade         : {record['relevance_grade']}")

        print("\nPASSAGE TEXT")
        print("-" * 100)
        print(record["text"])
        print("-" * 100)


if __name__ == "__main__":
    main()
