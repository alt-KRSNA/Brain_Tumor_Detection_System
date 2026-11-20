from typing import Any


class ContextBuilder:
    """
    Builds a compact, structured context from retrieved passages.

    Features:
    - Deduplication
    - Relevance-based ordering
    - Context budget management
    - Source metadata preservation
    """

    def __init__(
        self,
        max_context_tokens: int = 3000,
        max_passage_tokens: int = 700,
    ):
        self.max_context_tokens = max_context_tokens
        self.max_passage_tokens = max_passage_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Approximate token count.

        This is a lightweight estimation, not an exact tokenizer.
        """
        return max(1, len(text.split()))

    @staticmethod
    def normalize_text(text: str) -> str:
        return " ".join(text.lower().split())

    def remove_duplicates(self, passages: list[dict]) -> list[dict]:
        unique_passages = []
        seen = set()

        for passage in passages:
            text = passage.get("text", "").strip()

            if not text:
                continue

            normalized = self.normalize_text(text)

            if normalized in seen:
                continue

            seen.add(normalized)
            unique_passages.append(passage)

        return unique_passages

    def sort_by_relevance(self, passages: list[dict]) -> list[dict]:
        return sorted(
            passages,
            key=lambda passage: passage.get("score", 0),
            reverse=True,
        )

    def build_context(
        self,
        passages: list[dict],
    ) -> dict[str, Any]:

        passages = self.remove_duplicates(passages)
        passages = self.sort_by_relevance(passages)

        selected_passages = []
        sources = []
        used_tokens = 0

        for passage in passages:

            text = passage.get("text", "").strip()

            if not text:
                continue

            metadata = passage.get("metadata", {})

            source = metadata.get(
                "source",
                metadata.get("filename", "Unknown source"),
            )

            page = metadata.get("page")

            token_count = self.estimate_tokens(text)

            if token_count > self.max_passage_tokens:
                words = text.split()
                text = " ".join(words[:self.max_passage_tokens])
                token_count = self.estimate_tokens(text)

            if used_tokens + token_count > self.max_context_tokens:
                continue

            formatted_passage = {
                "text": text,
                "source": source,
                "page": page,
                "score": passage.get("score", 0),
                "estimated_tokens": token_count,
            }

            selected_passages.append(formatted_passage)

            source_reference = {
                "source": source,
                "page": page,
            }

            if source_reference not in sources:
                sources.append(source_reference)

            used_tokens += token_count

        context_text = "\n\n".join(
            (
                f"[Source: {passage['source']}, "
                f"Page: {passage['page']}]\n"
                f"{passage['text']}"
            )
            for passage in selected_passages
        )

        return {
            "context": context_text,
            "passages": selected_passages,
            "sources": sources,
            "passage_count": len(selected_passages),
            "estimated_tokens": used_tokens,
            "max_context_tokens": self.max_context_tokens,
        }