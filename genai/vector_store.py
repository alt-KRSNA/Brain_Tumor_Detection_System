
from pathlib import Path

import chromadb

from genai.document_loader import load_pdf_documents
from genai.chunking import split_documents
from genai.embeddings import generate_embeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

COLLECTION_NAME = "brain_tumor_knowledge"


def get_vector_collection():
    """
    Create or connect to a persistent ChromaDB collection.
    """

    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def store_embeddings(embedded_chunks):
    """
    Store text, embeddings and metadata in ChromaDB.
    """

    if not embedded_chunks:
        raise ValueError("No embeddings provided.")

    collection = get_vector_collection()

    ids = [
        f"chunk_{index}"
        for index in range(len(embedded_chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=[
            chunk["text"]
            for chunk in embedded_chunks
        ],
        embeddings=[
            chunk["embedding"].tolist()
            for chunk in embedded_chunks
        ],
        metadatas=[
            chunk["metadata"]
            for chunk in embedded_chunks
        ]
    )

    return collection


if __name__ == "__main__":

    print("=" * 60)
    print("BRAIN TUMOR DETECTION — VECTOR DATABASE")
    print("=" * 60)

    print("\nLoading medical documents...")

    documents = load_pdf_documents()

    print("\nCreating text chunks...")

    chunks = split_documents(documents)

    print("\nGenerating embeddings...")

    embedded_chunks = generate_embeddings(chunks)

    print("\nStoring embeddings in ChromaDB...")

    collection = store_embeddings(embedded_chunks)

    print("\nVECTOR DATABASE SUMMARY")
    print("-" * 60)

    print("Collection:", COLLECTION_NAME)
    print("Stored chunks:", collection.count())
    print("Database path:", VECTORSTORE_DIR)

    print("=" * 60)
