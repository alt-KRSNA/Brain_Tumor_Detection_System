
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = (
    PROJECT_ROOT / "knowledge_base" / "medical_documents"
)


def load_pdf_documents():
    """
    Load PDF files and extract text page by page.

    Returns:
        A list of dictionaries containing:
        - text
        - source filename
        - page number
    """

    documents = []

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF documents found in the medical documents folder."
        )

    for pdf_path in pdf_files:

        print(f"\nProcessing: {pdf_path.name}")

        try:
            reader = PdfReader(str(pdf_path))

            extracted_pages = 0

            for page_number, page in enumerate(reader.pages, start=1):

                text = page.extract_text() or ""
                text = text.strip()

                if not text:
                    print(
                        f"  Warning: No text on page {page_number}"
                    )
                    continue

                documents.append({
                    "text": text,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_number
                    }
                })

                extracted_pages += 1

            print(
                f"  Extracted {extracted_pages} pages"
            )

        except Exception as error:
            print(
                f"  Failed to process {pdf_path.name}: {error}"
            )

    return documents


if __name__ == "__main__":

    print("=" * 60)
    print("BRAIN TUMOR DETECTION — PDF DOCUMENT LOADER")
    print("=" * 60)

    docs = load_pdf_documents()

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)

    print(f"Total extracted pages: {len(docs)}")

    if docs:
        print("\nSample extracted content:")
        print("-" * 60)
        print(docs[0]["text"][:1000])

        print("\nSource:", docs[0]["metadata"]["source"])
        print("Page:", docs[0]["metadata"]["page"])

    print("=" * 60)
