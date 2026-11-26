import json
from pathlib import Path

from genai.hybrid_retriever import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "hybrid_retrieval"
)

CANDIDATE_FILE = REPORT_DIR / "hybrid_passage_candidates.json"

QUESTIONS = {
    "Q01": "What are the characteristics of glioma?",
    "Q02": "What are the characteristics of meningioma?",
    "Q03": "What are the characteristics of pituitary tumors?",
    "Q04": "What is a brain MRI used for?",
    "Q05": "What are common symptoms of brain tumors?",
}


def generate_candidates():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    retriever = HybridRetriever(final_top_k=5)

    all_candidates = []

    print("=" * 80)
    print("HYBRID PASSAGE CANDIDATE GENERATOR")
    print("=" * 80)

    for question_id, question in QUESTIONS.items():
        print(f"\n{question_id}: {question}")

        results = retriever.retrieve(query=question, final_top_k=5)

        for rank, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}

            candidate = {
                "question_id": question_id,
                "question": question,
                "rank": rank,
                "passage_id": result["id"],
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", "Unknown"),
                "text": result.get("text", ""),
                "dense_rank": result.get("dense_rank"),
                "bm25_rank": result.get("bm25_rank"),
                "dense_score": result.get("dense_score"),
                "bm25_score": result.get("bm25_score"),
                "rrf_score": result.get("rrf_score"),
            }

            all_candidates.append(candidate)

            print(
                f"  Rank {rank} | "
                f"{candidate['source']} | "
                f"Page {candidate['page']} | "
                f"ID {candidate['passage_id']}"
            )

    with CANDIDATE_FILE.open("w", encoding="utf-8") as file:
        json.dump(all_candidates, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Total candidates: {len(all_candidates)}")
    print(f"Saved: {CANDIDATE_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    generate_candidates()