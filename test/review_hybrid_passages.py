import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "hybrid_retrieval"
)

CANDIDATE_FILE = REPORT_DIR / "hybrid_passage_candidates.json"

ANNOTATION_FILE = (
    REPORT_DIR / "hybrid_relevance_annotations.csv"
)

FIELDNAMES = [
    "configuration",
    "question_id",
    "rank",
    "passage_id",
    "source",
    "page",
    "passage_text",
    "relevance_grade",
    "review_note",
    "review_confidence",
]

CONFIGURATION = "Hybrid"

QUESTIONS = {
    "Q01": "What are the characteristics of glioma?",
    "Q02": "What are the characteristics of meningioma?",
    "Q03": "What are the characteristics of pituitary tumors?",
    "Q04": "What is a brain MRI used for?",
    "Q05": "What are common symptoms of brain tumors?",
}


def load_candidates():
    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            "Candidate file not found. Run:\n"
            "python -m test.generate_hybrid_passage_candidates"
        )

    with CANDIDATE_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_annotations():
    if not ANNOTATION_FILE.exists():
        return {}

    annotations = {}

    with ANNOTATION_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            key = (
                row["question_id"],
                int(row["rank"]),
            )
            annotations[key] = row

    return annotations


def save_annotations(annotations):
    rows = sorted(
        annotations.values(),
        key=lambda row: (
            row["question_id"],
            int(row["rank"]),
        ),
    )

    with ANNOTATION_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(rows)


def display_passage(candidate):
    print("\n" + "=" * 100)

    print(
        f"{candidate['question_id']} | "
        f"Rank {candidate['rank']} | "
        f"{candidate['source']} | "
        f"Page {candidate['page']}"
    )

    print(f"Passage ID: {candidate['passage_id']}")

    print("-" * 100)

    print("QUESTION:")
    print(candidate["question"])

    print("-" * 100)

    print("PASSAGE:")
    print(candidate["text"])

    print("-" * 100)

    print("GRADE:")
    print("0 = Irrelevant")
    print("1 = Related but insufficient to answer")
    print("2 = Directly useful for answering")

    print("Enter 0, 1, 2, skip, or q to quit.")


def review():
    candidates = load_candidates()
    annotations = load_annotations()

    total = len(candidates)

    reviewed = sum(
        1
        for row in annotations.values()
        if row.get("relevance_grade", "").strip() in {"0", "1", "2"}
    )

    print("=" * 80)
    print("HYBRID PASSAGE RELEVENCE REVIEW")
    print("=" * 80)
    print(f"Total candidates: {total}")
    print(f"Already reviewed: {reviewed}")
    print(f"Remaining: {total - reviewed}")

    for candidate in candidates:
        key = (
            candidate["question_id"],
            int(candidate["rank"]),
        )

        existing = annotations.get(key)

        if existing and existing.get("relevance_grade", "").strip() in {
            "0", "1", "2"
        }:
            continue

        display_passage(candidate)

        while True:
            grade_input = input("\nYour grade: ").strip().lower()

            if grade_input == "q":
                save_annotations(annotations)

                print("\nReview stopped.")
                print(f"Saved: {ANNOTATION_FILE}")
                return

            if grade_input in {"skip", "s"}:
                print("Skipped. You can review this passage later.")
                break

            if grade_input not in {"0", "1", "2"}:
                print("Invalid input. Enter 0, 1, 2, skip, or q.")
                continue

            grade = grade_input
            note = input("Review note: ").strip()

            if not note:
                print("A review note is required.")
                continue

            confidence = input(
                "Confidence (high/medium/low): "
            ).strip().lower()

            if confidence not in {"high", "medium", "low"}:
                print("Invalid confidence.")
                continue

            row = {
                "configuration": CONFIGURATION,
                "question_id": candidate["question_id"],
                "rank": str(candidate["rank"]),
                "passage_id": candidate["passage_id"],
                "source": candidate["source"],
                "page": str(candidate["page"]),
                "passage_text": candidate["text"],
                "relevance_grade": grade,
                "review_note": note,
                "review_confidence": confidence,
            }

            annotations[key] = row

            save_annotations(annotations)

            reviewed += 1

            print(
                f"\nSaved annotation "
                f"({reviewed}/{total})"
            )

            break

    save_annotations(annotations)

    print("\n" + "=" * 80)
    print("REVIEW SESSION FINISHED")
    print(f"Reviewed: {reviewed}/{total}")
    print(f"Annotations: {ANNOTATION_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    review()