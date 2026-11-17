from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = PROJECT_ROOT / "vectorstore" / "experiments"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTIONS = [
    "experiment_chunking_a",
    "experiment_chunking_b",
    "experiment_chunking_c",
]

QUESTIONS = [
    "What is a glioma brain tumor?",
    "What is a pituitary tumor?",
    "What is a brain MRI used for?",
    "What are symptoms of a brain tumor?",
]


def main():

    print("=" * 75)
    print("PASSAGE-LEVEL RETRIEVAL COMPARISON")
    print("=" * 75)

    client = chromadb.PersistentClient(
        path=str(EXPERIMENT_DIR)
    )

    model = SentenceTransformer(MODEL_NAME)

    for question in QUESTIONS:

        print("\n" + "#" * 75)
        print("QUESTION:", question)
        print("#" * 75)

        query_embedding = model.encode(
            [question],
            convert_to_numpy=True
        )

        for collection_name in COLLECTIONS:

            collection = client.get_collection(
                name=collection_name
            )

            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=3,
                include=[
                    "documents",
                    "metadatas",
                    "distances"
                ]
            )

            print("\nCOLLECTION:", collection_name)

            for rank, (document, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ),
                start=1
            ):

                print(f"\nRANK: {rank}")
                print("Source:", metadata["source"])
                print("Page:", metadata["page"])
                print("Distance:", round(distance, 4))
                print("Passage:", document[:450].replace("\n", " "))
                print("-" * 60)


if __name__ == "__main__":
    main()
