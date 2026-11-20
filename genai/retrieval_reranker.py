import re


class RetrievalReranker:
    """
    Lightweight lexical reranker for retrieved passages.

    This is an experimental second-stage ranking component.
    It does not replace semantic retrieval.
    """

    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were",
        "what", "which", "who", "how", "why", "when",
        "where", "does", "do", "did", "can", "could",
        "would", "should", "of", "in", "on", "at", "to",
        "for", "with", "and", "or", "by", "from", "about",
        "this", "that", "these", "those", "it", "its",
        "me", "my", "tell", "explain", "describe",
    }

    def __init__(self, lexical_weight: float = 0.35):
        if not 0 <= lexical_weight <= 1:
            raise ValueError("lexical_weight must be between 0 and 1")

        self.lexical_weight = lexical_weight

    @staticmethod
    def tokenize(text: str) -> set[str]:
        return set(
            re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        )

    def extract_query_terms(self, query: str) -> set[str]:
        tokens = self.tokenize(query)

        return {
            token for token in tokens
            if token not in self.STOPWORDS and len(token) > 1
        }

    def calculate_lexical_score(
        self,
        query_terms: set[str],
        passage_text: str,
    ) -> float:
        passage_terms = self.tokenize(passage_text)

        if not query_terms or not passage_terms:
            return 0.0

        matched_terms = query_terms.intersection(passage_terms)

        return len(matched_terms) / len(query_terms)

    def rerank(
        self,
        query: str,
        passages: list[dict],
    ) -> list[dict]:

        query_terms = self.extract_query_terms(query)

        if not passages:
            return []

        scored_passages = []

        for index, passage in enumerate(passages):
            text = passage.get("text", "")

            lexical_score = self.calculate_lexical_score(
                query_terms,
                text,
            )

            # Preserve the original retrieval score.
            original_score = passage.get("score", 0.0)

            # Lexical ranking is used as a secondary signal.
            rerank_score = lexical_score

            result = passage.copy()
            result["lexical_score"] = lexical_score
            result["rerank_score"] = rerank_score
            result["original_rank"] = index + 1

            scored_passages.append(result)

        scored_passages.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return scored_passages