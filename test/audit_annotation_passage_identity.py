from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "Dense": ROOT / "test/reports/retrieval_baseline/baseline_relevance_annotations.csv",
    "Hybrid": ROOT / "test/reports/hybrid_retrieval/hybrid_relevance_annotations.csv",
}

def main():
    for name, path in FILES.items():
        df = pd.read_csv(path)

        passage_col = "chunk_id" if name == "Dense" else "passage_id"
        text_col = "text" if name == "Dense" else "passage_text"

        print("\n" + "=" * 100)
        print(f"{name.upper()} PASSAGE IDENTITY AUDIT")
        print("=" * 100)

        duplicated = df[df.duplicated(
            subset=["question_id", passage_col],
            keep=False
        )]

        if duplicated.empty:
            print("No duplicate question-passage IDs.")
            continue

        for (qid, pid), group in duplicated.groupby(
            ["question_id", passage_col]
        ):
            unique_texts = group[text_col].fillna("").nunique()

            print(f"\nQuestion: {qid} | Passage ID: {pid}")
            print(f"Records: {len(group)}")
            print(f"Distinct passage texts: {unique_texts}")

            for _, row in group.iterrows():
                print("-" * 80)
                print(f"Rank: {row.get('rank')}")
                print(f"Source: {row.get('source')}")
                print(f"Page: {row.get('page')}")
                print(f"Grade: {row.get('relevance_grade')}")
                print(f"Confidence: {row.get('review_confidence')}")
                print(f"Note: {row.get('review_note')}")
                print(f"Text:\n{row[text_col]}")

if __name__ == "__main__":
    main()