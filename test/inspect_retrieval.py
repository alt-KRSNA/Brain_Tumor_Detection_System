
from genai.retriever import retrieve_relevant_chunks


questions = [
    "What is a glioma brain tumor?",
    "What is a meningioma?",
    "What is a pituitary tumor?",
    "What is a brain MRI used for?",
    "What are the symptoms of a brain tumor?"
]


for question in questions:

    print("\n" + "=" * 90)
    print("QUESTION:", question)
    print("=" * 90)

    results = retrieve_relevant_chunks(
        query=question,
        top_k=3
    )

    for index, result in enumerate(results, start=1):

        metadata = result["metadata"]

        print(f"\nRESULT {index}")
        print("-" * 50)

        print("Source:", metadata["source"])
        print("Page:", metadata["page"])
        print("Distance:", round(result["distance"], 4))

        print("\nRetrieved text:")
        print(result["text"][:700])

        print("\n" + "-" * 90)
