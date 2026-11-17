from genai.retriever import retrieve_relevant_chunks


# ============================================================
# 1. EVALUATION DATASET
# ============================================================

evaluation_dataset = [
    {
        "question": "What is a glioma brain tumor?",
        "relevant_sources": ["glioma.pdf"],
        "expected_keywords": [
            "glial cells",
            "brain",
            "tumor"
        ]
    },
    {
        "question": "What is a meningioma?",
        "relevant_sources": ["meningioma.pdf", "glioma.pdf"],
        "expected_keywords": [
            "meninges",
            "membranes",
            "brain"
        ]
    },
    {
        "question": "What is a pituitary tumor?",
        "relevant_sources": [
            "pituitary_overview.pdf",
            "pituitary_diagnosis.pdf",
            "pituitary_symptoms.pdf"
        ],
        "expected_keywords": [
            "pituitary gland",
            "tumor",
            "hormone"
        ]
    },
    {
        "question": "What is a brain MRI used for?",
        "relevant_sources": [
            "brain_mri_classification.pdf",
            "glioma.pdf"
        ],
        "expected_keywords": [
            "magnetic resonance imaging",
            "MRI",
            "brain"
        ]
    },
    {
        "question": "What are the symptoms of a brain tumor?",
        "relevant_sources": [
            "glioma.pdf",
            "meningioma.pdf"
        ],
        "expected_keywords": [
            "headache",
            "seizures",
            "symptoms"
        ]
    }
]


# ============================================================
# 2. EVALUATION CONFIGURATION
# ============================================================

TOP_K = 3

total_questions = len(evaluation_dataset)

hit_count = 0
reciprocal_rank_sum = 0.0
precision_sum = 0.0
evidence_hit_count = 0


# ============================================================
# 3. EVALUATION
# ============================================================

print("=" * 80)
print("RAG RETRIEVAL AND EVIDENCE EVALUATION")
print("=" * 80)


for item in evaluation_dataset:

    question = item["question"]
    relevant_sources = item["relevant_sources"]
    expected_keywords = item["expected_keywords"]

    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    results = retrieve_relevant_chunks(
        query=question,
        top_k=TOP_K
    )

    relevant_ranks = []
    relevant_result_count = 0
    evidence_found = False

    for rank, result in enumerate(results, start=1):

        metadata = result["metadata"]
        source = metadata["source"]
        page = metadata["page"]
        retrieved_text = result["text"]

        is_relevant = source in relevant_sources

        if is_relevant:
            relevant_ranks.append(rank)
            relevant_result_count += 1

        normalized_text = retrieved_text.lower()

        matched_keywords = [
            keyword
            for keyword in expected_keywords
            if keyword.lower() in normalized_text
        ]

        if is_relevant and matched_keywords:
            evidence_found = True

        print(f"\nRank: {rank}")
        print("Source:", source)
        print("Page:", page)
        print("Distance:", round(result["distance"], 4))
        print("Relevant source:", is_relevant)
        print("Matched keywords:", matched_keywords)

        print("\nRetrieved evidence:")
        print(retrieved_text[:400])

    # --------------------------------------------------------
    # Hit@K and MRR
    # --------------------------------------------------------

    if relevant_ranks:

        hit_count += 1

        first_relevant_rank = min(relevant_ranks)

        reciprocal_rank_sum += 1 / first_relevant_rank

    else:
        first_relevant_rank = None

    # --------------------------------------------------------
    # Precision@K
    # --------------------------------------------------------

    precision_at_k = relevant_result_count / TOP_K

    precision_sum += precision_at_k

    # --------------------------------------------------------
    # Evidence keyword evaluation
    # --------------------------------------------------------

    if evidence_found:
        evidence_hit_count += 1

    print("\nQUESTION EVALUATION")
    print("-" * 40)

    print("First relevant rank:", first_relevant_rank)
    print(f"Precision@{TOP_K}: {precision_at_k:.2f}")
    print("Evidence keyword match:", evidence_found)


# ============================================================
# 4. FINAL METRICS
# ============================================================

hit_at_k = hit_count / total_questions

mrr = reciprocal_rank_sum / total_questions

mean_precision_at_k = precision_sum / total_questions

evidence_coverage = evidence_hit_count / total_questions


# ============================================================
# 5. FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("FINAL EVALUATION REPORT")
print("=" * 80)

print("Total questions:", total_questions)

print(f"Hit@{TOP_K}: {hit_at_k:.2f}")

print(f"MRR: {mrr:.4f}")

print(
    f"Mean Precision@{TOP_K}: "
    f"{mean_precision_at_k:.4f}"
)

print(
    "Evidence Keyword Coverage: "
    f"{evidence_coverage:.2f}"
)

print("\n" + "=" * 80)
print("EVALUATION COMPLETED")
print("=" * 80)
