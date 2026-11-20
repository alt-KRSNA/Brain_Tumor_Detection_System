import re
from pathlib import Path


class ResponseValidator:
    """
    Validates citation presence and citation-source matching.

    Expected citation format:
    [Source: filename.pdf, Page: 12]
    """

    CITATION_PATTERN = re.compile(
        r"\[Source:\s*(.*?),\s*Page:\s*(\d+)\s*\]",
        re.IGNORECASE,
    )

    @staticmethod
    def normalize_source(source: str) -> str:
        return Path(source.strip()).name.lower()

    def extract_citations(self, answer: str) -> list[dict]:
        citations = []

        for match in self.CITATION_PATTERN.finditer(answer):
            citations.append({
                "source": match.group(1).strip(),
                "page": int(match.group(2)),
            })

        return citations

    def validate(
        self,
        answer: str,
        retrieved_sources: list[dict],
    ) -> dict:

        citations = self.extract_citations(answer)

        if not citations:
            return {
                "valid": False,
                "status": "missing_citations",
                "citations": [],
                "unsupported_citations": [],
                "message": "No valid citations were found.",
            }

        available_sources = {
            (
                self.normalize_source(source.get("source", "")),
                source.get("page"),
            )
            for source in retrieved_sources
        }

        unsupported = []

        for citation in citations:
            citation_key = (
                self.normalize_source(citation["source"]),
                citation["page"],
            )

            if citation_key not in available_sources:
                unsupported.append(citation)

        if unsupported:
            return {
                "valid": False,
                "status": "unsupported_citations",
                "citations": citations,
                "unsupported_citations": unsupported,
                "message": (
                    "Some citations do not match the retrieved sources."
                ),
            }

        return {
            "valid": True,
            "status": "valid",
            "citations": citations,
            "unsupported_citations": [],
            "message": "All citations match retrieved sources.",
        }