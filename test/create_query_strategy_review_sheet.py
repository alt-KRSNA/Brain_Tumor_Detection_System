
"""
Create Query Strategy Relevance Review Sheet
===========================================

Creates a pooled review dataset from the top-10 candidate
results of all query strategies.

Relevance grades:
0 = Irrelevant
1 = Related but insufficient
2 = Directly useful

Existing annotations are not overwritten.
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "query_strategy_experiment"
    / "query_strategy_candidates.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "query_strategy_experiment"
    / "query_strategy_review.csv"
)


def main():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            "Candidate CSV not found. Run "
            "experiment_query_strategies.py first."
        )

    df = pd.read_csv(INPUT_CSV)

    # Pool all passages retrieved by the three strategies
    pooled = (
        df.groupby(
            ["question_id", "passage_id"],
            as_index=False,
        )
        .agg(
            best_rank=("rank", "min"),
            strategies=("strategy", lambda x: " | ".join(sorted(set(x)))),
            retrieved_count=("strategy", "nunique"),
        )
    )

    # Human-review columns
    pooled["relevance_grade"] = ""
    pooled["review_notes"] = ""

    pooled = pooled.sort_values(
        ["question_id", "best_rank", "passage_id"]
    )

    pooled.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("=" * 80)
    print("QUERY STRATEGY RELEVANCE REVIEW SHEET")
    print("=" * 80)

    print(f"Unique question-passage pairs: {len(pooled)}")
    print(f"Questions: {pooled['question_id'].nunique()}")
    print(f"Output: {OUTPUT_CSV}")

    print("\nGrade passages using:")
    print("0 = Irrelevant")
    print("1 = Related but insufficient")
    print("2 = Directly useful")

    print("\nReview sheet created successfully.")


if __name__ == "__main__":
    main()
