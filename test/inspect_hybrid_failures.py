
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports" / "hybrid_retrieval"

dense_path = BASE_DIR / "reports" / "retrieval_baseline" / "baseline_relevance_annotations.csv"
hybrid_path = REPORT_DIR / "hybrid_relevance_annotations.csv"

dense = pd.read_csv(dense_path)
hybrid = pd.read_csv(hybrid_path)

print("DENSE COLUMNS:", dense.columns.tolist())
print("HYBRID COLUMNS:", hybrid.columns.tolist())

# Select Dense configuration A
dense = dense[
    dense["configuration"].astype(str).str.upper().isin(
        ["A", "A - BASELINE", "BASELINE"]
    )
]

def inspect_question(qid, df, method):
    subset = df[df["question_id"].astype(str) == qid].sort_values("rank")

    print(f"\n{'-' * 40} {method} {'-' * 40}")

    for _, row in subset.iterrows():
        print(f"\nRank: {row.get('rank', 'N/A')}")
        print(f"Grade: {row.get('relevance_grade', 'N/A')}")
        print(f"Source: {row.get('source', 'N/A')}")
        print(f"Page: {row.get('page', 'N/A')}")

        # Handle different CSV schemas safely
        passage_id = row.get(
            "passage_id",
            row.get("chunk_id", row.get("id", "Not available"))
        )
        print(f"Passage ID: {passage_id}")

        passage_text = row.get(
            "passage_text",
            row.get("text", "Passage text unavailable")
        )
        print(f"Passage:\n{str(passage_text)[:1200]}")
        print("-" * 80)


for qid in ["Q01", "Q05"]:
    print("\n" + "=" * 100)
    print(f"QUESTION: {qid}")
    print("=" * 100)

    inspect_question(qid, dense, "DENSE")
    inspect_question(qid, hybrid, "HYBRID")
