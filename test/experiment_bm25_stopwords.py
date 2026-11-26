
from genai.hybrid_retriever import HybridRetriever, tokenize


STOPWORDS = {
    "what",
    "are",
    "the",
    "of",
    "is",
    "a",
    "an",
    "do",
    "does",
    "how",
    "why",
    "when",
    "where",
    "which",
    "who",
    "to",
    "in",
    "on",
    "for",
    "and",
}


QUESTIONS = {
    "Q01": {
        "query": "What are the characteristics of glioma?",
        "targets": ["chunk_31", "chunk_32", "chunk_33", "chunk_96"],
    },
    "Q05": {
        "query": "What are common symptoms of brain tumors?",
        "targets": ["chunk_1", "chunk_26", "chunk_41", "chunk_42", "chunk_96"],
    },
}


def get_ranked_results(retriever, query, top_k=10):

    scores = retriever.bm25.score(query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []

    for index in ranked_indices:

        if scores[index] <= 0:
            continue

        results.append({
            "id": retriever.ids[index],
            "score": scores[index],
        })

        if len(results) == top_k:
            break

    return results


def print_results(label, results, targets):

    print(f"\n{label}")
    print("-" * 80)

    target_ranks = {}

    for rank, result in enumerate(results, start=1):

        passage_id = result["id"]

        print(
            f"Rank {rank:2} | "
            f"{passage_id:<12} | "
            f"Score: {result['score']:.6f}"
        )

        if passage_id in targets:
            target_ranks[passage_id] = rank

    print("\nTarget passage positions:")

    for passage_id in targets:
        print(
            f"{passage_id:<12} -> "
            f"{target_ranks.get(passage_id, 'Not in top 10')}"
        )


def main():

    retriever = HybridRetriever()

    for question_id, details in QUESTIONS.items():

        original_query = details["query"]

        filtered_tokens = [
            token
            for token in tokenize(original_query)
            if token not in STOPWORDS
        ]

        filtered_query = " ".join(filtered_tokens)

        print("\n" + "=" * 85)
        print(f"{question_id}: {original_query}")
        print("=" * 85)

        print(f"Original tokens: {tokenize(original_query)}")
        print(f"Filtered tokens: {filtered_tokens}")

        original_results = get_ranked_results(
            retriever,
            original_query,
        )

        filtered_results = get_ranked_results(
            retriever,
            filtered_query,
        )

        print_results(
            "A - ORIGINAL BM25",
            original_results,
            details["targets"],
        )

        print_results(
            "B - FILTERED BM25",
            filtered_results,
            details["targets"],
        )


if __name__ == "__main__":
    main()
