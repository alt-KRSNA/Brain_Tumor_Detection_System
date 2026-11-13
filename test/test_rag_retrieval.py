
from genai.retriever import retrieve_relevant_chunks


evaluation_dataset = [
    {
        "question": "What is a glioma brain tumor?",
        "relevant_sources": ["glioma.pdf"]
    },
    {
        "question": "What is a meningioma?",
        "relevant_sources": ["meningioma.pdf", "glioma.pdf"]
    },
    {
        "question": "What is a pituitary tumor?",
        "relevant_sources": [
            "pituitary_overview.pdf",
            "pituitary_diagnosis.pdf",
            "pituitary_symptoms.pdf"
        ]
    },
    {
        "question": "What is a brain MRI used for?",
        "relevant_sources": [
            "brain_mri_classification.pdf",
            "glioma.pdf"
        ]
    },
    {
        "question": "What are the symptoms of a brain tumor?",
        "relevant_sources": [
            "glioma.pdf",
            "meningioma.pdf"
        ]
    }
]


total_hits = 0
reciprocal_rank_sum = 0


print("=" * 80)
print("RAG RETRIEVAL EVALUATION")
print("=" * 80)


for item in evaluation_dataset:

    question = item["question"]
    relevant_sources = item["relevant_sources"]

    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    results = retrieve_relevant_chunks(
        query=question,
        top_k=3
    )

    first_relevant_rank = None

    for rank, result in enumerate(results, start=1):

        metadata = result["metadata"]
        source = metadata["source"]

        print(f"\nRank: {rank}")
        print("Source:", source)
        print("Page:", metadata["page"])
        print("Distance:", round(result["distance"], 4))

        if (
            first_relevant_rank is None
            and source in relevant_sources
        ):
            first_relevant_rank = rank

    if first_relevant_rank is not None:

        total_hits += 1
        reciprocal_rank_sum += 1 / first_relevant_rank

        print("\nEvaluation: RELEVANT SOURCE FOUND")
        print("First relevant rank:", first_relevant_rank)

    else:
        print("\nEvaluation: NO EXPECTED SOURCE FOUND")


hit_at_3 = total_hits / len(evaluation_dataset)

mrr = reciprocal_rank_sum / len(evaluation_dataset)


print("\n" + "=" * 80)
print("FINAL EVALUATION RESULTS")
print("=" * 80)

print(f"Total questions: {len(evaluation_dataset)}")
print(f"Relevant source hits: {total_hits}")
print(f"Hit@3: {hit_at_3:.2f}")
print(f"MRR: {mrr:.4f}")

print("=" * 80)
