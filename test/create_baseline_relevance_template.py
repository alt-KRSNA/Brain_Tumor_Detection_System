from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "retrieval_baseline"
)

SOURCE_CSV = REPORT_DIR / "baseline_passages.csv"

OUTPUT_CSV = (
    REPORT_DIR
    / "baseline_relevance_annotations.csv"
)


ANNOTATION_FIELDS = [
    "relevance_grade",
    "review_note",
    "review_confidence",
]


def create_relevance_template(source_csv, output_csv):
    """
    Create a passage-level relevance annotation CSV.

    Grades:
        0 = Irrelevant
        1 = Related, but does not adequately answer
        2 = Directly useful for answering

    Existing annotation files are never overwritten.
    """

    source_csv = Path(source_csv)
    output_csv = Path(output_csv)

    if not source_csv.exists():
        raise FileNotFoundError(
            f"Baseline passage CSV not found: {source_csv}"
        )

    if output_csv.exists():
        raise FileExistsError(
            f"Annotation file already exists: {output_csv}\n"
            "It will not be overwritten to protect your annotations."
        )

    with source_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        source_fields = reader.fieldnames

        if not source_fields:
            raise ValueError(
                "Source CSV has no header."
            )

        required_fields = {
            "configuration",
            "question_id",
            "question",
            "expected_sources",
            "rank",
            "chunk_id",
            "source",
            "page",
            "distance",
            "text",
        }

        missing_fields = (
            required_fields - set(source_fields)
        )

        if missing_fields:
            raise ValueError(
                "Missing required source CSV fields: "
                f"{sorted(missing_fields)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(
            "Source CSV contains no passage records."
        )

    output_fields = (
        source_fields + ANNOTATION_FIELDS
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=output_fields,
        )

        writer.writeheader()

        for row in rows:
            row.update(
                {
                    "relevance_grade": "",
                    "review_note": "",
                    "review_confidence": "",
                }
            )

            writer.writerow(row)

    print("=" * 70)
    print("BASELINE RELEVANCE TEMPLATE CREATED")
    print("=" * 70)

    print("Source records :", len(rows))
    print("Output file   :", output_csv)
    print("Reviewed      : 0")
    print("Unreviewed    :", len(rows))

    print("\nRelevance grades:")
    print("0 = Irrelevant")
    print("1 = Related but insufficient to answer")
    print("2 = Directly useful for answering")

    return len(rows)


def main():
    create_relevance_template(
        SOURCE_CSV,
        OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()