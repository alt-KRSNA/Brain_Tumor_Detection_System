
import re
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = (
    PROJECT_ROOT / "knowledge_base" / "medical_documents"
)


def clean_extracted_text(text: str) -> str:
    """
    Clean common PDF extraction noise while preserving
    meaningful medical information.
    """

    if not text:
        return ""

    # Normalize whitespace
    text = text.replace("\xa0", " ")
    text = text.replace("\r", "\n")

    # Remove standalone URLs
    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    # Remove standalone page indicators such as 1/30 or 2/5
    text = re.sub(
        r"^\s*\d+\s*/\s*\d+\s*$",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove standalone numeric page numbers
    text = re.sub(
        r"^\s*\d+\s*$",
        "",
        text,
        flags=re.MULTILINE
    )

    # Normalize excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_pdf_documents():
    """
    Load PDF files and extract cleaned text page by page.

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
            skipped_pages = 0

            for page_number, page in enumerate(reader.pages, start=1):

                raw_text = page.extract_text() or ""

                text = clean_extracted_text(raw_text)

                if not text:
                    print(
                        f"  Warning: No useful text on page {page_number}"
                    )
                    skipped_pages += 1
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
                f"  Extracted pages: {extracted_pages}"
            )

            if skipped_pages:
                print(
                    f"  Skipped pages: {skipped_pages}"
                )

        except Exception as error:
            print(
                f"  Failed to process {pdf_path.name}: {error}"
            )

    return documents


if __name__ == "__main__":

    print("=" * 60)
    print("BRAIN TUMOR DETECTION - PDF DOCUMENT LOADER")
    print("=" * 60)

    docs = load_pdf_documents()

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)

    print(f"Total extracted pages: {len(docs)}")

    if docs:
        print("\nSample cleaned content:")
        print("-" * 60)

        print(docs[0]["text"][:1000])

        print("\nSource:", docs[0]["metadata"]["source"])
        print("Page:", docs[0]["metadata"]["page"])

    print("=" * 60)
