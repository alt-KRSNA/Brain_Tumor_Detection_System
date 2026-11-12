
from sentence_transformers import SentenceTransformer

from genai.chunking import (
    load_pdf_documents,
    split_documents
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def generate_embeddings(chunks):
    """
    Convert text chunks into numerical embeddings.
    """

    if not chunks:
        raise ValueError("No text chunks provided.")

    model = SentenceTransformer(MODEL_NAME)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):

        embedded_chunks.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": embedding
        })

    return embedded_chunks


if __name__ == "__main__":

    print("=" * 60)
    print("BRAIN TUMOR DETECTION — EMBEDDING GENERATION")
    print("=" * 60)

    documents = load_pdf_documents()

    chunks = split_documents(documents)

    embedded_chunks = generate_embeddings(chunks)

    print("\nEMBEDDING SUMMARY")
    print("-" * 60)

    print("Total chunks:", len(embedded_chunks))
    print("Embedding dimensions:", len(
        embedded_chunks[0]["embedding"]
    ))

    print("\nSample metadata:")
    print(embedded_chunks[0]["metadata"])

    print("\nSample embedding:")
    print(embedded_chunks[0]["embedding"][:5])

    print("=" * 60)
