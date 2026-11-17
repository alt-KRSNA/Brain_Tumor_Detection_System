from genai.document_loader import load_pdf_documents
from genai.chunking import split_documents


CONFIGURATIONS = [
    {"name": "A - Baseline", "chunk_size": 1000, "chunk_overlap": 200},
    {"name": "B - Medium", "chunk_size": 700, "chunk_overlap": 150},
    {"name": "C - Small", "chunk_size": 500, "chunk_overlap": 100},
]


def main():

    print("=" * 70)
    print("CHUNKING CONFIGURATION EXPERIMENT")
    print("=" * 70)

    documents = load_pdf_documents()

    print(f"\nTotal extracted pages: {len(documents)}")

    for config in CONFIGURATIONS:

        chunks = split_documents(
            documents,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )

        print("\n" + "-" * 70)
        print(config["name"])
        print("-" * 70)

        print("Chunk size:", config["chunk_size"])
        print("Chunk overlap:", config["chunk_overlap"])
        print("Total chunks:", len(chunks))

        if chunks:
            average_length = sum(
                len(chunk["text"]) for chunk in chunks
            ) / len(chunks)

            print("Average chunk length:", round(average_length, 2))

            print("\nSample chunk:")
            print(chunks[0]["text"][:300])

            print("\nSample metadata:")
            print(chunks[0]["metadata"])

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
