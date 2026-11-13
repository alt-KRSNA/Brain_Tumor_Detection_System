
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


def reset_vector_collection():
    """
    Delete the existing collection and create a fresh one.
    This prevents outdated chunks from remaining in the database.
    """

    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR)
    )

    existing_collections = client.list_collections()

    collection_names = [
        collection.name
        for collection in existing_collections
    ]

    if COLLECTION_NAME in collection_names:
        client.delete_collection(name=COLLECTION_NAME)
        print("Old vector collection deleted.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("Fresh vector collection created.")

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
    print("BRAIN TUMOR DETECTION - VECTOR DATABASE")
    print("=" * 60)

    print("\nLoading medical documents...")

    documents = load_pdf_documents()

    print(f"\nTotal extracted pages: {len(documents)}")

    print("\nCreating text chunks...")

    chunks = split_documents(documents)

    print(f"Total generated chunks: {len(chunks)}")

    print("\nGenerating embeddings...")

    embedded_chunks = generate_embeddings(chunks)

    print(f"Total generated embeddings: {len(embedded_chunks)}")

    print("\nResetting old vector database...")

    reset_vector_collection()

    print("\nStoring new embeddings in ChromaDB...")

    collection = store_embeddings(embedded_chunks)

    stored_count = collection.count()

    print("\nVECTOR DATABASE SUMMARY")
    print("-" * 60)

    print("Collection:", COLLECTION_NAME)
    print("Extracted pages:", len(documents))
    print("Generated chunks:", len(chunks))
    print("Generated embeddings:", len(embedded_chunks))
    print("Stored chunks:", stored_count)
    print("Database path:", VECTORSTORE_DIR)

    if stored_count != len(embedded_chunks):
        raise RuntimeError(
            "Vector database verification failed: "
            "stored chunk count does not match generated embeddings."
        )

    print("\nVector database rebuilt and verified successfully.")

    print("=" * 60)
