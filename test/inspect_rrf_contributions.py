
import pandas as pd
from pathlib import Path

from genai.hybrid_retriever import (
    HybridRetriever,
    reciprocal_rank_fusion,
)

BASE_DIR = Path(__file__).resolve().parent

ANNOTATION_PATH = (
    BASE_DIR / "reports" / "hybrid_retrieval"
    / "hybrid_relevance_annotations.csv"
)

QUESTIONS = {
    "Q01": "What are the characteristics of glioma?",
    "Q05": "What are common symptoms of brain tumors?",
}


def main():

    annotations = pd.read_csv(ANNOTATION_PATH)

    retriever = HybridRetriever(
        dense_top_k=10,
        bm25_top_k=10,
        final_top_k=5,
    )

    for question_id, query in QUESTIONS.items():

        print("\n" + "=" * 100)
        print(f"{question_id}: {query}")
        print("=" * 100)

        dense_results = retriever.dense_retrieve(query)

        bm25_results = retriever.bm25_retrieve(query)

        fused_results = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
        )

        print("\nDENSE RANKING")
        print("-" * 100)

        for result in dense_results:
            print(
                f"Rank: {result['dense_rank']:2} | "
                f"ID: {result['id']} | "
                f"Distance: {result['distance']:.5f}"
            )

        print("\nBM25 RANKING")
        print("-" * 100)

        for result in bm25_results:
            print(
                f"Rank: {result['bm25_rank']:2} | "
                f"ID: {result['id']} | "
                f"Score: {result['bm25_score']:.5f}"
            )

        print("\nFUSED RANKING")
        print("-" * 100)

        for final_rank, result in enumerate(
            fused_results[:10],
            start=1,
        ):

            passage_id = result["id"]

            matching = annotations[
                (annotations["question_id"] == question_id)
                & (
                    annotations["passage_id"].astype(str)
                    == str(passage_id)
                )
            ]

            grade = (
                matching.iloc[0]["relevance_grade"]
                if not matching.empty
                else "NOT ANNOTATED"
            )

            print(
                f"\nFinal rank: {final_rank}"
                f"\nPassage: {passage_id}"
                f"\nDense rank: {result['dense_rank']}"
                f"\nBM25 rank: {result['bm25_rank']}"
                f"\nDense distance: {result['dense_score']}"
                f"\nBM25 score: {result['bm25_score']}"
                f"\nRRF score: {result['rrf_score']:.8f}"
                f"\nRelevance grade: {grade}"
            )

            print(f"Text: {result['text'][:350]}")
            print("-" * 70)


if __name__ == "__main__":
    main()
