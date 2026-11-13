
from genai.retriever import retrieve_relevant_chunks
from genai.llm_service import generate_response


RAG_SYSTEM_PROMPT = """
You are an educational Brain Tumor Information Assistant.

Your job is to answer questions using ONLY the medical
context provided by the retrieval system.

Rules:
1. Use the supplied context as your primary and only
   source of medical facts.
2. Do not invent medical facts or references.
3. If the context does not contain enough information,
   clearly say that the available knowledge base does
   not provide sufficient information.
4. Explain medical concepts in simple language.
5. Include source filenames and page numbers when
   supporting information is available.
6. Never diagnose a person or interpret an MRI as a
   medical diagnosis.
7. Do not recommend treatments for an individual.
8. Remind users that this system is educational and
   cannot replace a qualified medical professional.
"""


def build_context(retrieved_chunks):
    """
    Combine retrieved medical chunks into one context.
    """

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):

        source = chunk["metadata"]["source"]
        page = chunk["metadata"]["page"]

        context_parts.append(
            f"""
[Source {index}]
Filename: {source}
Page: {page}

Content:
{chunk["text"]}
"""
        )

    return "\n\n".join(context_parts)


def extract_sources(retrieved_chunks):
    """
    Extract unique source filenames and page numbers
    from retrieved chunks.
    """

    sources = []
    seen_sources = set()

    for chunk in retrieved_chunks:

        metadata = chunk["metadata"]

        source = metadata["source"]
        page = metadata["page"]

        source_key = (source, page)

        if source_key not in seen_sources:

            seen_sources.add(source_key)

            sources.append({
                "filename": source,
                "page": page
            })

    return sources


def ask_medical_question(question: str, top_k: int = 3):
    """
    Retrieve medical evidence and generate an answer
    using Groq.

    Returns:
        dict: Contains generated answer and source references.
    """

    # Validate question
    if not question or not question.strip():

        raise ValueError("Question cannot be empty.")

    # Step 1: Retrieve relevant medical information
    retrieved_chunks = retrieve_relevant_chunks(
        query=question,
        top_k=top_k
    )

    # Step 2: Handle empty retrieval
    if not retrieved_chunks:

        return {
            "answer": (
                "I could not find relevant information in "
                "the medical knowledge base."
            ),
            "sources": []
        }

    # Step 3: Build context
    context = build_context(retrieved_chunks)

    # Step 4: Extract source references
    sources = extract_sources(retrieved_chunks)

    # Step 5: Create user prompt
    user_prompt = f"""
Answer the following question using the medical
context provided below.

USER QUESTION:
{question}

RETRIEVED MEDICAL CONTEXT:
{context}

Provide a clear, structured and educational answer.

Include source filenames and page numbers.
Do not introduce medical facts that are absent
from the supplied context.
"""

    # Step 6: Generate answer using Groq
    answer = generate_response(
        user_prompt=user_prompt,
        system_prompt=RAG_SYSTEM_PROMPT,
        temperature=0.2,
        max_tokens=1024
    )

    # Step 7: Return structured response
    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":

    question = "What is a glioma brain tumor?"

    print("=" * 60)
    print("BRAIN TUMOR DETECTION — RAG PIPELINE")
    print("=" * 60)

    print("\nQuestion:", question)
    print("\nRetrieving evidence and generating answer...\n")

    try:

        result = ask_medical_question(question)

        print("-" * 60)
        print("AI GENERATED ANSWER")
        print("-" * 60)

        print(result["answer"])

        print("\n" + "-" * 60)
        print("SOURCE REFERENCES")
        print("-" * 60)

        for source in result["sources"]:

            print(
                f"Source: {source['filename']} "
                f"| Page: {source['page']}"
            )

    except Exception as error:

        print("\nRAG pipeline failed:", error)

    print("\n" + "=" * 60)
