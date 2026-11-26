
"""
Inspect Dense, BM25, and Hybrid retrieval rankings.

Usage:
    python -m test.inspect_hybrid_retrieval

    python -m test.inspect_hybrid_retrieval "What are glioma symptoms?"
"""

import sys

from genai.hybrid_retriever import HybridRetriever


def print_results(
    title: str,
    results: list[dict],
    limit: int = 5,
):

    print("\n" + "=" * 100)

    print(title.upper())

    print("=" * 100)

    if not results:

        print("No results found.")

        return

    for rank, result in enumerate(
        results[:limit],
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        print("\n" + "-" * 100)

        print(f"Rank: {rank}")

        print(f"ID: {result.get('id')}")

        print(
            f"Source: {metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page: {metadata.get('page', 'Unknown')}"
        )

        if result.get("dense_rank") is not None:

            print(
                f"Dense rank: {result['dense_rank']}"
            )

        if result.get("bm25_rank") is not None:

            print(
                f"BM25 rank: {result['bm25_rank']}"
            )

        if result.get("dense_score") is not None:

            print(
                f"Dense score/distance: "
                f"{result['dense_score']:.6f}"
            )

        if result.get("bm25_score") is not None:

            print(
                f"BM25 score: "
                f"{result['bm25_score']:.6f}"
            )

        if result.get("rrf_score") is not None:

            print(
                f"RRF score: "
                f"{result['rrf_score']:.6f}"
            )

        print("\nPASSAGE:")

        print(
            result.get("text", "")
        )


def main():

    if len(sys.argv) > 1:

        query = " ".join(sys.argv[1:])

    else:

        query = input(
            "Enter your retrieval question: "
        ).strip()

    if not query:

        print("Question cannot be empty.")

        return

    retriever = HybridRetriever(
        dense_top_k=10,
        bm25_top_k=10,
        final_top_k=5,
    )

    report = retriever.retrieve_with_details(
        query=query,
        final_top_k=5,
    )

    print("\nQUERY:")

    print(query)

    print_results(
        "Dense Retrieval",
        report["dense_results"],
    )

    print_results(
        "BM25 Retrieval",
        report["bm25_results"],
    )

    print_results(
        "Hybrid Retrieval - RRF",
        report["hybrid_results"],
    )


if __name__ == "__main__":
    main()
