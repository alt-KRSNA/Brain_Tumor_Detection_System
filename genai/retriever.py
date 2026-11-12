
from sentence_transformers import SentenceTransformer

from genai.vector_store import get_vector_collection
from genai.embeddings import MODEL_NAME


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 3
):
    """
    Retrieve the most relevant medical text chunks
    for a given user question.
    """

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Convert user question into an embedding
    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    # Connect to ChromaDB
    collection = get_vector_collection()

    if collection.count() == 0:
        raise ValueError("Knowledge base is empty.")

    # Search for similar chunks
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved_chunks = []

    for index in range(len(results["documents"][0])):

        retrieved_chunks.append({
            "text": results["documents"][0][index],
            "metadata": results["metadatas"][0][index],
            "distance": results["distances"][0][index]
        })

    return retrieved_chunks


if __name__ == "__main__":

    question = "What is a glioma brain tumor?"

    print("=" * 60)
    print("BRAIN TUMOR DETECTION — SEMANTIC RETRIEVAL")
    print("=" * 60)

    results = retrieve_relevant_chunks(
        query=question,
        top_k=3
    )

    print("\nQUESTION:", question)

    for index, result in enumerate(results, start=1):

        print("\n" + "-" * 60)
        print(f"RESULT {index}")
        print("-" * 60)

        print("Source:", result["metadata"]["source"])
        print("Page:", result["metadata"]["page"])
        print("Distance:", round(result["distance"], 4))

        print("\nRetrieved text:")
        print(result["text"][:700])

    print("\n" + "=" * 60)

