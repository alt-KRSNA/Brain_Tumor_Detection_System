
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

dense_path = (
    BASE_DIR / "reports" / "retrieval_baseline"
    / "baseline_relevance_annotations.csv"
)

hybrid_path = (
    BASE_DIR / "reports" / "hybrid_retrieval"
    / "hybrid_relevance_annotations.csv"
)

dense = pd.read_csv(dense_path)
hybrid = pd.read_csv(hybrid_path)

# Dense configuration A
dense = dense[
    dense["configuration"].astype(str).str.upper().isin(
        ["A", "A - BASELINE", "BASELINE"]
    )
].copy()

# Standardize passage identifiers
dense["passage_id"] = dense["chunk_id"].astype(str)
hybrid["passage_id"] = hybrid["passage_id"].astype(str)

keys = ["question_id", "passage_id"]

comparison = dense.merge(
    hybrid,
    on=keys,
    how="inner",
    suffixes=("_dense", "_hybrid")
)

print("=" * 90)
print("RELEVANCE ANNOTATION CONSISTENCY AUDIT")
print("=" * 90)

print(f"\nOverlapping passages: {len(comparison)}")

mismatches = comparison[
    comparison["relevance_grade_dense"]
    != comparison["relevance_grade_hybrid"]
]

if mismatches.empty:
    print("\nNo inconsistent labels found.")
else:
    print(f"\nInconsistent labels: {len(mismatches)}")

    for _, row in mismatches.iterrows():
        print("\n" + "-" * 80)
        print(f"Question: {row['question_id']}")
        print(f"Passage: {row['passage_id']}")
        print(f"Source: {row['source_dense']}")
        print(f"Dense grade: {row['relevance_grade_dense']}")
        print(f"Hybrid grade: {row['relevance_grade_hybrid']}")
        print(f"\nPassage:\n{row['text'][:800]}")
        print(f"\nDense note: {row['review_note_dense']}")
        print(f"Hybrid note: {row['review_note_hybrid']}")

print("\nAudit completed. No files were modified.")
