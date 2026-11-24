from pathlib import Path

from test.retrieval_evidence import load_retrieval_evidence
from test.failure_cause_template import generate_failure_cause_template


def main():
    records = load_retrieval_evidence()

    output_path = (
        Path(__file__).parent
        / "reports"
        / "failure_cause_annotations.csv"
    )

    count = generate_failure_cause_template(
        records,
        output_path
    )

    print("=" * 60)
    print("FAILURE CAUSE ANNOTATION TEMPLATE GENERATED")
    print("=" * 60)
    print(f"Total retrieval records : {len(records)}")
    print(f"Passages for review    : {count}")
    print(f"Output file            : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()