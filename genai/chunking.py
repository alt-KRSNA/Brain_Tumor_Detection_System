
from langchain_text_splitters import RecursiveCharacterTextSplitter

from genai.document_loader import load_pdf_documents


def split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
):
    """
    Split extracted documents into smaller text chunks.

    Each chunk retains its original source and page metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for document in documents:

        text_chunks = splitter.split_text(document["text"])

        for chunk in text_chunks:

            chunks.append({
                "text": chunk,
                "metadata": document["metadata"].copy()
            })

    return chunks


if __name__ == "__main__":

    print("=" * 60)
    print("NOTE: BRAIN TUMOR KNOWLEDGE BASE — CHUNKING")
    print("=" * 60)

    documents = load_pdf_documents()

    chunks = split_documents(documents)

    print("\nCHUNKING SUMMARY")
    print("-" * 60)

    print(f"Extracted pages : {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    if chunks:

        print("\nSample chunk:")
        print("-" * 60)

        print(chunks[0]["text"])

        print("\nMetadata:")
        print(chunks[0]["metadata"])

    print("=" * 60)
