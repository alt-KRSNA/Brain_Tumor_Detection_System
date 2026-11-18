
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENT_DIR = PROJECT_ROOT / "vectorstore" / "experiments"
OUTPUT_FILE = PROJECT_ROOT / "test" / "passage_candidates.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5

CONFIGURATIONS = {
    "A": "experiment_chunking_a",
    "B": "experiment_chunking_b",
    "C": "experiment_chunking_c",
}

QUESTIONS = {
    "glioma": "What is a glioma and how is it defined?",
    "meningioma": "What is a meningioma?",
    "pituitary": "What is a pituitary tumor?",
    "mri": "What is brain MRI and what is it used for?",
    "symptoms": "What are the common symptoms of brain tumors?",
}


def main():

    print("=" * 75)
    print("PASSAGE-LEVEL EVALUATION: CANDIDATE GENERATION")
    print("=" * 75)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"\n{OUTPUT_FILE.name} already exists.\n"
            "To protect existing annotations, this script will not overwrite it."
        )

    if not EXPERIMENT_DIR.exists():
        raise FileNotFoundError(
            f"Experiment database not found: {EXPERIMENT_DIR}"
        )

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(
        path=str(EXPERIMENT_DIR)
    )

    candidates = {
        "instructions": {
            "relevance_grades": {
                "0": "Irrelevant passage",
                "1": "Partially relevant or supporting passage",
                "2": "Directly answers the question"
            },
            "note": (
                "Manually assign relevance_grade for each passage. "
                "Do not judge relevance only by the PDF filename."
            )
        },
        "configurations": {}
    }

    for config_name, collection_name in CONFIGURATIONS.items():

        print(f"\n{'-' * 75}")
        print(f"CONFIGURATION {config_name}: {collection_name}")
        print("-" * 75)

        try:
            collection = client.get_collection(
                name=collection_name
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load collection: {collection_name}"
            ) from error

        config_results = {}

        for question_id, question in QUESTIONS.items():

            print(f"\nQuestion: {question}")

            query_embedding = model.encode(
                question,
                convert_to_numpy=True
            )

            results = collection.query(
                query_embeddings=[
                    query_embedding.tolist()
                ],
                n_results=TOP_K,
                include=[
                    "documents",
                    "metadatas",
                    "distances"
                ]
            )

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            passages = []

            for rank, (chunk_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances),
                start=1
            ):

                passage = {
                    "chunk_id": chunk_id,
                    "rank": rank,
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page"),
                    "distance": round(float(distance), 6),
                    "text": document,
                    "relevance_grade": None
                }

                passages.append(passage)

                print(
                    f"Rank {rank}: "
                    f"{passage['source']} | "
                    f"Page {passage['page']} | "
                    f"Distance {passage['distance']}"
                )

            config_results[question_id] = {
                "question": question,
                "passages": passages
            }

        candidates["configurations"][config_name] = config_results

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            candidates,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 75)
    print("CANDIDATE GENERATION COMPLETED")
    print("=" * 75)

    print(f"\nOutput file: {OUTPUT_FILE}")
    print("Configurations:", len(CONFIGURATIONS))
    print("Questions:", len(QUESTIONS))
    print("Top passages per question:", TOP_K)
    print(
        "Total passages:",
        len(CONFIGURATIONS) * len(QUESTIONS) * TOP_K
    )

    print("\nNext: manually assign relevance grades (0, 1, or 2).")


if __name__ == "__main__":
    main()
