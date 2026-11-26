
from genai.hybrid_retriever import HybridRetriever, tokenize


QUERY = "What are common symptoms of brain tumors?"

TARGETS = [
    "chunk_1",
    "chunk_26",
    "chunk_41",
    "chunk_42",
    "chunk_96",
]

TERM_GROUPS = {
    "Original": "what are common symptoms of brain tumors",
    "Without what/are": "common symptoms of brain tumors",
    "Without common": "what are symptoms of brain tumors",
    "Without brain": "what are common symptoms of tumors",
    "Without tumors": "what are common symptoms of brain",
    "Intent only": "symptoms",
    "Topic + intent": "symptoms brain tumors",
}


def evaluate(retriever, label, query):

    scores = retriever.bm25.score(query)

    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    print(f"\n{label}: {query}")
    print("-" * 75)

    ranks = {}

    for index in ranked:

        if scores[index] <= 0:
            continue

        passage_id = retriever.ids[index]

        if passage_id in TARGETS:
            ranks[passage_id] = (
                ranked.index(index) + 1,
                scores[index],
            )

    for passage_id in TARGETS:

        if passage_id in ranks:
            rank, score = ranks[passage_id]
            print(
                f"{passage_id:<12} Rank: {rank:<4} "
                f"Score: {score:.6f}"
            )
        else:
            print(f"{passage_id:<12} Not retrieved")


def main():

    retriever = HybridRetriever()

    print(f"Original question: {QUERY}")

    for label, query in TERM_GROUPS.items():
        evaluate(retriever, label, query)


if __name__ == "__main__":
    main()
