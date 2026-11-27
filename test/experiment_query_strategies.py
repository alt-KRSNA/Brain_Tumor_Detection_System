
"""
Query Strategy Experiment
========================

Phase 5 - Hybrid Retrieval Optimization

Compares BM25 retrieval rankings for:
A. Original benchmark queries
B. Stop-word filtered queries
C. Intent-focused queries

Does NOT modify:
- Production retriever
- ChromaDB
- Relevance annotations
- Existing evaluation reports

Outputs:
- CSV with top-10 results for all 15 conditions
- Console ranking comparison for target passages
"""

from pathlib import Path
import csv

from genai.hybrid_retriever import HybridRetriever, tokenize


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "test"
    / "reports"
    / "query_strategy_experiment"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "query_strategy_candidates.csv"


# ============================================================
# BENCHMARK QUESTIONS
# ============================================================

QUESTIONS = {
    "Q01": {
        "original": "What are the characteristics of glioma?",
        "intent": "glioma characteristics",
        "targets": ["chunk_31", "chunk_32", "chunk_33", "chunk_96"],
    },
    "Q02": {
        "original": "What are the characteristics of meningioma?",
        "intent": "meningioma characteristics",
        "targets": ["chunk_97"],
    },
    "Q03": {
        "original": "What are the characteristics of pituitary tumor?",
        "intent": "pituitary tumor characteristics",
        "targets": ["chunk_114", "chunk_115"],
    },
    "Q04": {
        "original": "What is a brain MRI used for?",
        "intent": "brain MRI purpose uses",
        "targets": ["chunk_107", "chunk_6"],
    },
    "Q05": {
        "original": "What are common symptoms of brain tumors?",
        "intent": "symptoms brain tumors",
        "targets": ["chunk_1", "chunk_26", "chunk_41", "chunk_42", "chunk_96"],
    },
}


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "what",
    "are",
    "the",
    "of",
    "is",
    "a",
    "an",
    "do",
    "does",
    "how",
    "why",
    "when",
    "where",
    "which",
    "who",
    "to",
    "in",
    "on",
    "for",
    "and",
    "used",
}


# ============================================================
# QUERY STRATEGIES
# ============================================================

def build_queries(question):

    original_query = question["original"]

    filtered_tokens = [
        token
        for token in tokenize(original_query)
        if token not in STOPWORDS
    ]

    stopword_filtered_query = " ".join(filtered_tokens)

    intent_query = question["intent"]

    return {
        "A_original": original_query,
        "B_stopword_filtered": stopword_filtered_query,
        "C_intent_focused": intent_query,
    }


# ============================================================
# BM25 RANKING
# ============================================================

def rank_bm25(retriever, query, top_k=10):

    scores = retriever.bm25.score(query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []

    for index in ranked_indices:

        score = scores[index]

        # Exclude passages with no lexical match
        if score <= 0:
            continue

        passage_id = retriever.ids[index]

        results.append({
            "passage_id": passage_id,
            "score": float(score),
        })

        if len(results) >= top_k:
            break

    return results


# ============================================================
# TARGET PASSAGE RANKING
# ============================================================

def find_target_ranks(results, targets):

    rank_map = {
        result["passage_id"]: rank
        for rank, result in enumerate(results, start=1)
    }

    return {
        target: rank_map.get(target)
        for target in targets
    }


# ============================================================
# EXPERIMENT
# ============================================================

def main():

    retriever = HybridRetriever()

    print("=" * 90)
    print("QUERY STRATEGY EXPERIMENT")
    print("=" * 90)

    print(f"Collection: {retriever.collection_name}")
    print(f"Corpus size: {len(retriever.ids)}")
    print("Strategies: Original | Stop-word filtered | Intent-focused")
    print("Retriever: BM25 only")
    print("Top-K: 10")

    all_rows = []

    # --------------------------------------------------------
    # Iterate benchmark questions
    # --------------------------------------------------------

    for question_id, question in QUESTIONS.items():

        queries = build_queries(question)

        print("\n" + "=" * 90)
        print(f"{question_id}: {question['original']}")
        print("=" * 90)

        for strategy, query in queries.items():

            results = rank_bm25(
                retriever,
                query,
                top_k=10,
            )

            target_ranks = find_target_ranks(
                results,
                question["targets"],
            )

            print(f"\n{strategy}")
            print(f"Query: {query}")
            print("-" * 75)

            for rank, result in enumerate(results, start=1):

                passage_id = result["passage_id"]

                is_target = passage_id in question["targets"]

                print(
                    f"Rank {rank:2} | "
                    f"{passage_id:<12} | "
                    f"Score: {result['score']:.6f}"
                    f"{' | TARGET' if is_target else ''}"
                )

                all_rows.append({
                    "question_id": question_id,
                    "strategy": strategy,
                    "query": query,
                    "rank": rank,
                    "passage_id": passage_id,
                    "bm25_score": result["score"],
                    "is_previous_target": is_target,
                })

            print("\nTracked passage ranks:")

            for target, rank in target_ranks.items():

                print(
                    f"{target:<12} -> "
                    f"{rank if rank is not None else 'Not in top 10'}"
                )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    fieldnames = [
        "question_id",
        "strategy",
        "query",
        "rank",
        "passage_id",
        "bm25_score",
        "is_previous_target",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_rows)


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 90)
    print("EXPERIMENT COMPLETED")
    print("=" * 90)

    print(f"Total ranking records: {len(all_rows)}")
    print(f"Questions: {len(QUESTIONS)}")
    print("Strategies: 3")
    print(f"Expected records: {len(QUESTIONS) * 3 * 10}")

    print(f"\nResults saved to:\n{OUTPUT_CSV}")

    print("\nNo production files or annotations were modified.")


if __name__ == "__main__":
    main()
