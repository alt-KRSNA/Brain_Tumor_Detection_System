
import math

from genai.hybrid_retriever import HybridRetriever, tokenize


QUESTIONS = {
    "Q01": {
        "query": "What are the characteristics of glioma?",
        "passages": [
            "chunk_31",
            "chunk_32",
            "chunk_33",
            "chunk_96",
        ],
    },
    "Q05": {
        "query": "What are common symptoms of brain tumors?",
        "passages": [
            "chunk_1",
            "chunk_26",
            "chunk_41",
            "chunk_42",
            "chunk_96",
        ],
    },
}


def explain_bm25(retriever, query, passage_id):

    bm25 = retriever.bm25

    try:
        index = retriever.ids.index(passage_id)
    except ValueError:
        print(f"Passage not found: {passage_id}")
        return

    query_tokens = tokenize(query)
    document_tokens = bm25.tokenized_documents[index]
    document_length = bm25.document_lengths[index]

    contributions = []

    for token in query_tokens:

        df = bm25.document_frequency.get(token, 0)

        if df == 0:
            continue

        idf = math.log(
            1 + (
                bm25.total_documents - df + 0.5
            ) / (df + 0.5)
        )

        tf = document_tokens.count(token)

        if tf == 0:
            continue

        denominator = tf + bm25.k1 * (
            1 - bm25.b
            + bm25.b * (
                document_length
                / bm25.average_document_length
            )
        )

        contribution = (
            idf * tf * (bm25.k1 + 1)
        ) / denominator

        contributions.append((token, tf, contribution))

    contributions.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    print(f"\nPassage: {passage_id}")
    print(f"Document length: {document_length}")
    print(f"Total BM25 score: {sum(x[2] for x in contributions):.6f}")

    if not contributions:
        print("No query terms matched.")
        return

    print("Matched query terms:")

    for token, tf, score in contributions:
        print(
            f"  {token:<15} "
            f"TF={tf:<3} "
            f"Contribution={score:.6f}"
        )


def main():

    retriever = HybridRetriever()

    for question_id, details in QUESTIONS.items():

        query = details["query"]

        print("\n" + "=" * 85)
        print(f"{question_id}: {query}")
        print("=" * 85)

        for passage_id in details["passages"]:
            explain_bm25(
                retriever,
                query,
                passage_id,
            )


if __name__ == "__main__":
    main()
