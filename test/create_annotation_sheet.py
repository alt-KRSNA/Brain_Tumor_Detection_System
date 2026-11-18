
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CANDIDATES_FILE = PROJECT_ROOT / "test" / "passage_candidates.json"
ANNOTATIONS_FILE = PROJECT_ROOT / "test" / "passage_annotations.json"
OUTPUT_FILE = PROJECT_ROOT / "test" / "passage_annotation_sheet.csv"


def main():
    with open(CANDIDATES_FILE, encoding="utf-8") as file:
        candidates = json.load(file)

    with open(ANNOTATIONS_FILE, encoding="utf-8") as file:
        annotations = json.load(file)

    rows = []

    for config_name, questions in candidates["configurations"].items():

        for question_id, question_data in questions.items():

            for passage in question_data["passages"]:

                rank = passage["rank"]
                annotation_key = f"{config_name}|{question_id}|{rank}"

                existing = annotations.get(annotation_key, {})

                rows.append({
                    "configuration": config_name,
                    "question_id": question_id,
                    "question": question_data["question"],
                    "rank": rank,
                    "source": passage["source"],
                    "page": passage["page"],
                    "distance": passage["distance"],
                    "chunk_id": passage["chunk_id"],
                    "passage": passage["text"],
                    "relevance_grade": existing.get(
                        "relevance_grade", ""
                    ),
                    "annotation_notes": ""
                })

    fieldnames = [
        "configuration",
        "question_id",
        "question",
        "rank",
        "source",
        "page",
        "distance",
        "chunk_id",
        "passage",
        "relevance_grade",
        "annotation_notes"
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    completed = sum(
        1 for row in rows
        if row["relevance_grade"] in (0, 1, 2)
    )

    print("=" * 70)
    print("PASSAGE ANNOTATION SHEET GENERATED")
    print("=" * 70)
    print(f"Total passages: {len(rows)}")
    print(f"Already annotated: {completed}")
    print(f"Remaining: {len(rows) - completed}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

