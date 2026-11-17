from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from genai.document_loader import load_pdf_documents
from genai.chunking import split_documents


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = PROJECT_ROOT / "vectorstore" / "experiments"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CONFIGURATIONS = [
    {"name": "A - Baseline", "size": 1000, "overlap": 200},
    {"name": "B - Medium", "size": 700, "overlap": 150},
    {"name": "C - Small", "size": 500, "overlap": 100},
]

QUESTIONS = [
    {
        "question": "What is a glioma brain tumor?",
        "expected_sources": ["glioma.pdf"],
        "keywords": ["glial cells", "brain", "tumor"],
    },
    {
        "question": "What is a meningioma?",
        "expected_sources": ["meningioma.pdf", "glioma.pdf"],
        "keywords": ["meninges", "membranes", "brain"],
    },
    {
        "question": "What is a pituitary tumor?",
        "expected_sources": [
            "pituitary_overview.pdf",
            "pituitary_diagnosis.pdf",
            "pituitary_symptoms.pdf",
        ],
        "keywords": ["pituitary gland", "tumor", "hormone"],
    },
    {
        "question": "What is a brain MRI used for?",
        "expected_sources": [
            "brain_mri_classification.pdf",
            "glioma.pdf",
        ],
        "keywords": ["magnetic resonance imaging", "MRI", "brain"],
    },
    {
        "question": "What are symptoms of a brain tumor?",
        "expected_sources": ["glioma.pdf", "meningioma.pdf"],
        "keywords": ["headache", "seizures", "symptoms"],
    },
]


def evaluate_configuration(config, documents, model, client):

    collection_name = (
        f"experiment_chunking_{config['name'][0].lower()}"
    )

    print("\n" + "=" * 75)
    print(config["name"])
    print("=" * 75)

    chunks = split_documents(
        documents,
        chunk_size=config["size"],
        chunk_overlap=config["overlap"],
    )

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Delete only this experiment's collection if it already exists.
    # Never touch the production collection.
    existing = [
        item.name for item in client.list_collections()
    ]

    if collection_name in existing:
        client.delete_collection(name=collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[chunk["metadata"] for chunk in chunks],
    )

    print("Collection:", collection_name)
    print("Total chunks:", len(chunks))

    hit_count = 0
    reciprocal_ranks = []
    precision_scores = []
    evidence_count = 0

    for item in QUESTIONS:

        query_embedding = model.encode(
            [item["question"]],
            convert_to_numpy=True,
        )

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_sources = results["metadatas"][0]
        retrieved_texts = results["documents"][0]
        distances = results["distances"][0]

        relevant_ranks = []

        for rank, metadata in enumerate(retrieved_sources, start=1):
            if metadata["source"] in item["expected_sources"]:
                relevant_ranks.append(rank)

        if relevant_ranks:
            hit_count += 1
            reciprocal_ranks.append(1 / relevant_ranks[0])
        else:
            reciprocal_ranks.append(0)

        relevant_count = sum(
            metadata["source"] in item["expected_sources"]
            for metadata in retrieved_sources
        )

        precision_scores.append(relevant_count / 3)

        all_retrieved_text = " ".join(retrieved_texts).lower()

        matched_keywords = [
            keyword
            for keyword in item["keywords"]
            if keyword.lower() in all_retrieved_text
        ]

        if matched_keywords:
            evidence_count += 1

        print("\nQUESTION:", item["question"])

        for rank, (metadata, text, distance) in enumerate(
            zip(retrieved_sources, retrieved_texts, distances),
            start=1,
        ):
            print(
                f"\n  Rank {rank}: {metadata['source']} "
                f"| Page {metadata['page']} "
                f"| Distance {distance:.4f}"
            )
            print("  Text:", text[:250].replace("\n", " "))

        print("\n  Matched keywords:", matched_keywords)

    total = len(QUESTIONS)

    metrics = {
        "hit_at_3": hit_count / total,
        "mrr": sum(reciprocal_ranks) / total,
        "precision_at_3": sum(precision_scores) / total,
        "keyword_coverage": evidence_count / total,
    }

    print("\nMETRICS")
    print("-" * 50)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics


def main():

    print("=" * 75)
    print("ISOLATED CHUNKING RETRIEVAL EXPERIMENT")
    print("=" * 75)

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(EXPERIMENT_DIR)
    )

    print("\nLoading medical documents...")
    documents = load_pdf_documents()

    print("Extracted pages:", len(documents))

    print("\nLoading embedding model once...")
    model = SentenceTransformer(MODEL_NAME)

    all_metrics = {}

    for config in CONFIGURATIONS:

        metrics = evaluate_configuration(
            config,
            documents,
            model,
            client,
        )

        all_metrics[config["name"]] = metrics

    print("\n" + "=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)

    print(
        f"{'Configuration':<20}"
        f"{'Hit@3':>10}"
        f"{'MRR':>10}"
        f"{'Precision@3':>15}"
        f"{'Keyword':>12}"
    )

    for name, metrics in all_metrics.items():

        print(
            f"{name:<20}"
            f"{metrics['hit_at_3']:>10.4f}"
            f"{metrics['mrr']:>10.4f}"
            f"{metrics['precision_at_3']:>15.4f}"
            f"{metrics['keyword_coverage']:>12.4f}"
        )

    print("\nProduction collection was not modified.")
    print("Experiment database:", EXPERIMENT_DIR)


if __name__ == "__main__":
    main()
