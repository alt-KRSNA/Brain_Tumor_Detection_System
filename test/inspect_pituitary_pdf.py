
from genai.document_loader import load_pdf_documents

documents = load_pdf_documents()

pituitary_documents = [
    document for document in documents
    if document["metadata"]["source"].startswith("pituitary_")
]

print("\n" + "=" * 75)
print("PITUITARY KNOWLEDGE BASE VALIDATION")
print("=" * 75)

for document in pituitary_documents:

    text = document["text"]

    print("\nSource:", document["metadata"]["source"])
    print("Page:", document["metadata"]["page"])
    print("Characters:", len(text))
    print("Preview:", text[:400].replace("\n", " "))

print("\n" + "=" * 75)
print("TOTAL PITUITARY PAGES:", len(pituitary_documents))
print("=" * 75)
