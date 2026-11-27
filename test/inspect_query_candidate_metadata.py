
from pathlib import Path
import pandas as pd

from genai.hybrid_retriever import HybridRetriever


ROOT = Path(__file__).resolve().parent.parent

REVIEW_FILE = (
    ROOT / "test/reports/query_strategy_experiment/query_strategy_review.csv"
)


def main():
    review_df = pd.read_csv(REVIEW_FILE)

    retriever = HybridRetriever()

    collection = retriever.collection

    ids = review_df["passage_id"].dropna().unique().tolist()

    print("=" * 90)
    print("CHROMADB CANDIDATE METADATA INSPECTION")
    print("=" * 90)

    print(f"Unique candidate IDs: {len(ids)}")
    print(f"Collection count: {collection.count()}")

    result = collection.get(
        ids=ids,
        include=["documents", "metadatas"],
    )

    for passage_id, document, metadata in zip(
        result["ids"],
        result["documents"],
        result["metadatas"],
    ):
        print("\n" + "-" * 90)
        print("Passage ID:", passage_id)
        print("Metadata:", metadata)
        print("Document:")
        print(document[:1200] if document else "No document")


if __name__ == "__main__":
    main()

