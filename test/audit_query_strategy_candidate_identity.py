
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "Query candidates": (
        ROOT / "test/reports/query_strategy_experiment/query_strategy_candidates.csv",
        "passage_id",
    ),
    "Review sheet": (
        ROOT / "test/reports/query_strategy_experiment/query_strategy_review.csv",
        "passage_id",
    ),
}

def main():
    for name, (path, passage_col) in FILES.items():
        df = pd.read_csv(path)

        print("\n" + "=" * 90)
        print(f"{name.upper()} IDENTITY AUDIT")
        print("=" * 90)

        if "source" not in df.columns or "page" not in df.columns:
            print("Source/page columns unavailable.")
            print("Columns:", list(df.columns))
            continue

        duplicated = df[
            df.duplicated(
                subset=["question_id", passage_col],
                keep=False,
            )
        ]

        print("Rows:", len(df))
        print("Duplicate question-passage records:", len(duplicated))

        for (qid, pid), group in duplicated.groupby(
            ["question_id", passage_col]
        ):
            print(f"\n{qid} | {pid}")
            print("Sources:", group["source"].unique().tolist())
            print("Pages:", group["page"].unique().tolist())
            print("Records:", len(group))

if __name__ == "__main__":
    main()
