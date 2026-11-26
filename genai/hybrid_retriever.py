
"""
Hybrid retrieval experiment.

Combines:
1. Dense retrieval from ChromaDB
2. BM25 lexical retrieval
3. Reciprocal Rank Fusion (RRF)

Production retriever remains untouched.
"""

import re
from collections import Counter
from typing import Any

import chromadb


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent

EXPERIMENT_VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "experiments"

DENSE_COLLECTION_NAME = "experiment_chunking_a"

DEFAULT_DENSE_TOP_K = 10
DEFAULT_BM25_TOP_K = 10
DEFAULT_FINAL_TOP_K = 5

RRF_K = 60


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """
    Tokenize text while preserving hyphenated/underscored terms.

    Example:
    brain-mri -> brain-mri
    """
    if not text:
        return []

    return re.findall(
        r"[a-z0-9]+(?:[-_][a-z0-9]+)*",
        text.lower()
    )


# ---------------------------------------------------------
# BM25 implementation
# ---------------------------------------------------------

class BM25:
    """
    A lightweight BM25 implementation.

    No external BM25 dependency required.
    """

    def __init__(
        self,
        documents: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = documents
        self.k1 = k1
        self.b = b

        self.tokenized_documents = [
            tokenize(document)
            for document in documents
        ]

        self.document_lengths = [
            len(tokens)
            for tokens in self.tokenized_documents
        ]

        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
            if self.document_lengths
            else 0.0
        )

        self.document_frequency = Counter()

        for tokens in self.tokenized_documents:
            unique_tokens = set(tokens)

            for token in unique_tokens:
                self.document_frequency[token] += 1

        self.total_documents = len(documents)

    def score(self, query: str) -> list[float]:
        """
        Calculate BM25 scores for all documents.
        """

        query_tokens = tokenize(query)

        if not query_tokens or not self.documents:
            return [0.0] * len(self.documents)

        scores = [0.0] * self.total_documents

        for token in query_tokens:

            df = self.document_frequency.get(token, 0)

            if df == 0:
                continue

            # BM25 inverse document frequency
            idf = (
                __import__("math").log(
                    1 + (
                        self.total_documents - df + 0.5
                    ) / (df + 0.5)
                )
            )

            for index, document_tokens in enumerate(
                self.tokenized_documents
            ):

                term_frequency = document_tokens.count(token)

                if term_frequency == 0:
                    continue

                document_length = self.document_lengths[index]

                denominator = (
                    term_frequency
                    + self.k1 * (
                        1 - self.b
                        + self.b * (
                            document_length
                            / self.average_document_length
                        )
                    )
                )

                score = idf * (
                    term_frequency * (self.k1 + 1)
                ) / denominator

                scores[index] += score

        return scores

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Return BM25-ranked documents.
        """

        scores = self.score(query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:top_k]:

            if scores[index] <= 0:
                continue

            results.append({
                "index": index,
                "text": self.documents[index],
                "bm25_score": scores[index],
            })

        return results


# ---------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------

def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    rrf_k: int = RRF_K,
) -> list[dict]:
    """
    Combine rankings using Reciprocal Rank Fusion.

    RRF score:
        1 / (rrf_k + rank)

    Ranks begin at 1.
    """

    fused_scores = {}
    result_lookup = {}

    rankings = [
        ("dense", dense_results),
        ("bm25", bm25_results),
    ]

    for method, results in rankings:

        for rank, result in enumerate(results, start=1):

            document_id = result["id"]

            if document_id not in fused_scores:
                fused_scores[document_id] = 0.0

                result_lookup[document_id] = {
                    **result,
                    "dense_rank": None,
                    "bm25_rank": None,
                    "dense_score": None,
                    "bm25_score": None,
                }

            fused_scores[document_id] += (
                1.0 / (rrf_k + rank)
            )

            if method == "dense":

                result_lookup[document_id]["dense_rank"] = rank

                result_lookup[document_id]["dense_score"] = (
                    result.get("dense_score")
                )

            elif method == "bm25":

                result_lookup[document_id]["bm25_rank"] = rank

                result_lookup[document_id]["bm25_score"] = (
                    result.get("bm25_score")
                )

    ranked_results = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    final_results = []

    for document_id, score in ranked_results:

        result = result_lookup[document_id].copy()

        result["rrf_score"] = score

        final_results.append(result)

    return final_results


# ---------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------

class HybridRetriever:
    """
    Dense + BM25 + RRF retriever.

    Uses the existing experimental ChromaDB collection
    experiment_chunking_a (baseline chunking: 1000/200).
    """

    def __init__(
        self,
        collection_name: str = DENSE_COLLECTION_NAME,
        dense_top_k: int = DEFAULT_DENSE_TOP_K,
        bm25_top_k: int = DEFAULT_BM25_TOP_K,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
    ):

        self.collection_name = collection_name

        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k

        self.client = chromadb.PersistentClient(
            path=str(EXPERIMENT_VECTORSTORE_DIR)
        )

        self.collection = self.client.get_collection(
            name=collection_name
        )

        # Load the full experimental corpus.
        corpus = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.documents = corpus["documents"] or []

        self.metadatas = corpus["metadatas"] or []

        self.ids = corpus["ids"]

        if not self.documents:
            raise ValueError(
                f"Collection '{collection_name}' is empty."
            )

        if len(self.documents) != len(self.ids):
            raise ValueError(
                "Document and ID counts do not match."
            )

        # Build BM25 index once.
        self.bm25 = BM25(self.documents)

        print("Hybrid Retriever initialized")
        print(f"Collection: {collection_name}")
        print(f"Corpus size: {len(self.documents)}")
        print(f"Dense top-k: {dense_top_k}")
        print(f"BM25 top-k: {bm25_top_k}")
        print(f"Final top-k: {final_top_k}")

    # -----------------------------------------------------
    # Dense retrieval
    # -----------------------------------------------------

    def dense_retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:

        top_k = top_k or self.dense_top_k

        top_k = min(top_k, self.collection.count())

        response = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        results = []

        returned_ids = response["ids"][0]

        returned_documents = response["documents"][0]

        returned_metadatas = response["metadatas"][0]

        returned_distances = response["distances"][0]

        for rank, (
            document_id,
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                returned_ids,
                returned_documents,
                returned_metadatas,
                returned_distances,
            ),
            start=1,
        ):

            results.append({
                "id": document_id,
                "text": document,
                "metadata": metadata or {},
                "dense_rank": rank,
                "dense_score": distance,
                "distance": distance,
            })

        return results

    # -----------------------------------------------------
    # BM25 retrieval
    # -----------------------------------------------------

    def bm25_retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:

        top_k = top_k or self.bm25_top_k

        bm25_results = self.bm25.retrieve(
            query=query,
            top_k=top_k,
        )

        results = []

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):

            index = result["index"]

            results.append({
                "id": self.ids[index],
                "text": self.documents[index],
                "metadata": self.metadatas[index] or {},
                "bm25_rank": rank,
                "bm25_score": result["bm25_score"],
            })

        return results

    # -----------------------------------------------------
    # Hybrid retrieval
    # -----------------------------------------------------

    def retrieve(
        self,
        query: str,
        final_top_k: int | None = None,
    ) -> list[dict]:

        final_top_k = final_top_k or self.final_top_k

        dense_results = self.dense_retrieve(
            query,
            top_k=self.dense_top_k,
        )

        bm25_results = self.bm25_retrieve(
            query,
            top_k=self.bm25_top_k,
        )

        fused_results = reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
        )

        return fused_results[:final_top_k]

    # -----------------------------------------------------
    # Complete retrieval report
    # -----------------------------------------------------

    def retrieve_with_details(
        self,
        query: str,
        final_top_k: int | None = None,
    ) -> dict:

        final_top_k = final_top_k or self.final_top_k

        dense_results = self.dense_retrieve(query)

        bm25_results = self.bm25_retrieve(query)

        hybrid_results = reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
        )[:final_top_k]

        return {
            "query": query,
            "dense_results": dense_results,
            "bm25_results": bm25_results,
            "hybrid_results": hybrid_results,
        }


# ---------------------------------------------------------
# CLI test
# ---------------------------------------------------------

def main():

    retriever = HybridRetriever()

    query = "What are the symptoms of glioma?"

    report = retriever.retrieve_with_details(query)

    for method in [
        "dense_results",
        "bm25_results",
        "hybrid_results",
    ]:

        print("\n" + "=" * 80)

        print(method.upper())

        print("=" * 80)

        for result in report[method]:

            metadata = result.get("metadata", {})

            print(
                f"\nSource: {metadata.get('source', 'Unknown')}"
            )

            print(
                f"Page: {metadata.get('page', 'Unknown')}"
            )

            print(
                f"ID: {result['id']}"
            )

            if result.get("rrf_score") is not None:

                print(
                    f"RRF score: {result['rrf_score']:.6f}"
                )

            if result.get("bm25_score") is not None:

                print(
                    f"BM25 score: {result['bm25_score']:.6f}"
                )

            if result.get("dense_score") is not None:

                print(
                    f"Dense score/distance: "
                    f"{result['dense_score']:.6f}"
                )

            print(result["text"][:700])


if __name__ == "__main__":
    main()
