# test/reuse_query_strategy_labels.py

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REVIEW_FILE = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "query_strategy_experiment"
    / "query_strategy_review.csv"
)

DENSE_FILE = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "retrieval_baseline"
    / "baseline_relevance_annotations.csv"
)

HYBRID_FILE = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "hybrid_retrieval"
    / "hybrid_relevance_annotations.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "query_strategy_experiment"
    / "query_strategy_review_with_existing_labels.csv"
)


def normalize_id(value):
    """Normalize IDs for reliable matching."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def extract_annotations(file_path, passage_column, configuration_column=None):
    """Read an annotation file and return question-passage-grade mappings."""

    df = pd.read_csv(file_path)

    required_columns = {
        "question_id",
        passage_column,
        "relevance_grade",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {file_path.name}: {missing}"
        )

    annotations = {}

    for _, row in df.iterrows():

        question_id = normalize_id(row["question_id"])
        passage_id = normalize_id(row[passage_column])
        grade = row["relevance_grade"]

        if not question_id or not passage_id or pd.isna(grade):
            continue

        grade = int(grade)

        if grade not in (0, 1, 2):
            raise ValueError(
                f"Invalid grade {grade} in {file_path.name}"
            )

        key = (question_id, passage_id)

        annotations.setdefault(key, set()).add(grade)

    return annotations


def main():

    print("=" * 80)
    print("REUSE EXISTING QUERY STRATEGY ANNOTATIONS")
    print("=" * 80)

    review_df = pd.read_csv(REVIEW_FILE)

# Prevent pandas from treating blank text columns as numeric.
    review_df["review_notes"] = review_df["review_notes"].astype("object")
    review_df["relevance_grade"] = review_df["relevance_grade"].astype("object")

    dense_annotations = extract_annotations(
        DENSE_FILE,
        passage_column="chunk_id",
    )

    hybrid_annotations = extract_annotations(
        HYBRID_FILE,
        passage_column="passage_id",
    )

    print(f"Review pairs: {len(review_df)}")
    print(f"Dense annotation pairs: {len(dense_annotations)}")
    print(f"Hybrid annotation pairs: {len(hybrid_annotations)}")

    reused_count = 0
    unjudged_count = 0
    conflict_count = 0

    review_df["existing_dense_grade"] = pd.NA
    review_df["existing_hybrid_grade"] = pd.NA
    review_df["label_source"] = ""
    review_df["label_conflict"] = False

    for index, row in review_df.iterrows():

        question_id = normalize_id(row["question_id"])
        passage_id = normalize_id(row["passage_id"])

        key = (question_id, passage_id)

        dense_grades = dense_annotations.get(key, set())
        hybrid_grades = hybrid_annotations.get(key, set())

        dense_grade = (
            next(iter(dense_grades))
            if len(dense_grades) == 1
            else None
        )

        hybrid_grade = (
            next(iter(hybrid_grades))
            if len(hybrid_grades) == 1
            else None
        )

        if dense_grade is not None:
            review_df.at[index, "existing_dense_grade"] = dense_grade

        if hybrid_grade is not None:
            review_df.at[index, "existing_hybrid_grade"] = hybrid_grade

        combined_grades = dense_grades | hybrid_grades

        sources = []

        if dense_grades:
            sources.append("Dense")

        if hybrid_grades:
            sources.append("Hybrid")

        review_df.at[index, "label_source"] = (
            " + ".join(sources) if sources else "None"
        )

        # Existing annotation files can themselves have duplicate
        # conflicting grades for the same question-passage pair.
        if len(dense_grades) > 1 or len(hybrid_grades) > 1:

            review_df.at[index, "label_conflict"] = True

            review_df.at[index, "review_notes"] = (
                "CONFLICT: duplicate grades inside existing annotation file. "
                f"Dense={sorted(dense_grades)}, "
                f"Hybrid={sorted(hybrid_grades)}"
            )

            conflict_count += 1
            continue

        # Conflict between Dense and Hybrid human judgments.
        if len(combined_grades) > 1:

            review_df.at[index, "label_conflict"] = True

            review_df.at[index, "review_notes"] = (
                "CONFLICT: Dense and Hybrid annotations disagree. "
                f"Dense={dense_grade}, Hybrid={hybrid_grade}. "
                "Manual resolution required."
            )

            conflict_count += 1
            continue

        # Reuse one consistent existing judgment.
        if len(combined_grades) == 1:

            reused_grade = next(iter(combined_grades))

            review_df.at[index, "relevance_grade"] = reused_grade

            review_df.at[index, "review_notes"] = (
                f"Reused existing human judgment. Source: "
                f"{', '.join(sources)}."
            )

            reused_count += 1

        else:

            # No prior judgment for this pair.
            review_df.at[index, "relevance_grade"] = pd.NA

            review_df.at[index, "review_notes"] = (
                "UNJUDGED: Manual relevance review required."
            )

            unjudged_count += 1

    # Preserve blank grades as blank in the output CSV.
    review_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 80)
    print("ANNOTATION REUSE SUMMARY")
    print("=" * 80)

    print(f"Existing labels reused: {reused_count}")
    print(f"Pairs requiring new judgment: {unjudged_count}")
    print(f"Conflicting pairs requiring review: {conflict_count}")

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("\nOriginal annotation files were NOT modified.")
    print("Original review sheet was NOT modified.")

    print("\nCompleted successfully.")


if __name__ == "__main__":
    main()