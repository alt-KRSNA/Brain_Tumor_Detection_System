
from pathlib import Path
import argparse
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "test/reports/query_strategy_experiment/"
    / "query_strategy_review_identity_verified.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "test/reports/query_strategy_experiment/"
    / "query_strategy_review_completed.csv"
)

QUESTIONS = {
    "Q01": "What are the characteristics of glioma?",
    "Q02": "What are the characteristics of meningioma?",
    "Q03": "What are the characteristics of pituitary tumor?",
    "Q04": "What is a brain MRI used for?",
    "Q05": "What are common symptoms of brain tumors?",
}


def save_review(df):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        choices=list(QUESTIONS.keys()),
        help="Review only one question, e.g. Q01",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum pending passages to review (0 = all)",
    )
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    # Resume from the previous review output if it exists.
    if OUTPUT_FILE.exists():
        df = pd.read_csv(OUTPUT_FILE)
        print("Resuming from existing review progress.")
    else:
        df = pd.read_csv(INPUT_FILE)

    required = {
        "question_id",
        "passage_id",
        "source",
        "page",
        "passage_text",
        "verified_relevance_grade",
        "identity_status",
        "review_notes",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    # Ensure blank review notes can accept text values.
    df["review_notes"] = df["review_notes"].fillna("").astype(object)
    df.loc[df["review_notes"].isna(), "review_notes"] = ""

    df["verified_relevance_grade"] = pd.to_numeric(
        df["verified_relevance_grade"], errors="coerce"
    )

    pending = df["verified_relevance_grade"].isna()

    if args.question:
        pending &= df["question_id"].eq(args.question)

    pending_indices = df.index[pending].tolist()

    if args.limit > 0:
        pending_indices = pending_indices[:args.limit]

    print("=" * 90)
    print("IDENTITY-VERIFIED QUERY STRATEGY PASSAGE REVIEW")
    print("=" * 90)
    print(f"Pending passages in this session: {len(pending_indices)}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    print("Grades:")
    print("0 = Irrelevant")
    print("1 = Related but insufficient to answer")
    print("2 = Directly useful to answer")
    print()
    print("Commands: s = skip | q = quit")
    print("=" * 90)

    reviewed_this_session = 0

    for index in pending_indices:
        row = df.loc[index]
        question_id = str(row["question_id"])

        print("\n" + "=" * 90)
        print(f"Question ID : {question_id}")
        print(f"Question    : {QUESTIONS.get(question_id, 'Unknown question')}")
        print(f"Passage ID  : {row['passage_id']}")
        print(f"Source      : {row['source']}")
        print(f"Page        : {row['page']}")
        print(f"Best rank   : {row['best_rank']}")
        print(f"Strategies  : {row['strategies']}")
        print("-" * 90)
        print(row["passage_text"])
        print("=" * 90)

        while True:
            choice = input("Relevance grade [0/1/2, s, q]: ").strip().lower()

            if choice == "q":
                save_review(df)
                print("\nReview stopped. Progress saved.")
                print(f"Reviewed this session: {reviewed_this_session}")
                return

            if choice == "s":
                print("Skipped. No grade assigned.")
                break

            if choice in {"0", "1", "2"}:
                grade = int(choice)
                notes = input("Review note (optional): ").strip()

                df.at[index, "verified_relevance_grade"] = grade
                df.at[index, "review_notes"] = notes
                df.at[index, "identity_status"] = "HUMAN_REVIEW_COMPLETED"

                save_review(df)
                reviewed_this_session += 1

                print(f"Saved grade {grade}.")
                break

            print("Invalid input. Enter 0, 1, 2, s, or q.")

    save_review(df)

    print("\n" + "=" * 90)
    print("REVIEW SESSION COMPLETED")
    print("=" * 90)
    print(f"Reviewed this session: {reviewed_this_session}")
    print(
        "Still pending:",
        int(df["verified_relevance_grade"].isna().sum()),
    )
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
