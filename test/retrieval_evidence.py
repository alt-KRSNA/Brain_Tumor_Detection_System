
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_FILE = PROJECT_ROOT / "test" / "passage_candidates.json"
ANNOTATIONS_FILE = PROJECT_ROOT / "test" / "passage_annotations.json"


class RetrievalEvidenceError(ValueError):
    """Raised when retrieval evidence or annotations are invalid."""


def load_json(path):
    """Load a JSON file and return its contents."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_annotations(annotations):
    """Validate the annotation dictionary and its records."""
    if not isinstance(annotations, dict):
        raise RetrievalEvidenceError(
            "Annotations must be a dictionary."
        )

    for key, record in annotations.items():
        if not isinstance(record, dict):
            raise RetrievalEvidenceError(
                f"Annotation {key!r} must contain a dictionary."
            )

        required = {
            "configuration",
            "question_id",
            "rank",
            "relevance_grade",
        }

        missing = required - record.keys()

        if missing:
            raise RetrievalEvidenceError(
                f"Annotation {key!r} is missing fields: {sorted(missing)}"
            )

        grade = record["relevance_grade"]

        if isinstance(grade, bool) or grade not in (0, 1, 2):
            raise RetrievalEvidenceError(
                f"Invalid relevance grade for {key!r}: {grade!r}"
            )

        expected_key = (
            f"{record['configuration']}|"
            f"{record['question_id']}|"
            f"{record['rank']}"
        )

        if key != expected_key:
            raise RetrievalEvidenceError(
                f"Annotation key mismatch: expected {expected_key!r}, "
                f"received {key!r}."
            )


def load_retrieval_evidence(
    candidates_path=CANDIDATES_FILE,
    annotations_path=ANNOTATIONS_FILE,
):
    """
    Join candidate passages with their relevance annotations.

    Returns:
        list[dict]: One evidence record per annotated passage.
    """
    candidates_data = load_json(candidates_path)
    annotations = load_json(annotations_path)

    validate_annotations(annotations)

    if not isinstance(candidates_data, dict):
        raise RetrievalEvidenceError(
            "Candidate data must be a dictionary."
        )

    configurations = candidates_data.get("configurations")

    if not isinstance(configurations, dict):
        raise RetrievalEvidenceError(
            "Candidate data must contain a configurations dictionary."
        )

    evidence = []
    matched_keys = set()

    for configuration, questions in configurations.items():
        for question_id, question_data in questions.items():
            passages = question_data.get("passages", [])

            for passage in passages:
                rank = passage.get("rank")

                annotation_key = (
                    f"{configuration}|{question_id}|{rank}"
                )

                if annotation_key not in annotations:
                    raise RetrievalEvidenceError(
                        f"Missing annotation for passage: {annotation_key}"
                    )

                annotation = annotations[annotation_key]
                matched_keys.add(annotation_key)

                if (
                    annotation["configuration"] != configuration
                    or annotation["question_id"] != question_id
                    or annotation["rank"] != rank
                ):
                    raise RetrievalEvidenceError(
                        f"Annotation metadata mismatch: {annotation_key}"
                    )

                required_passage_fields = {
                    "chunk_id",
                    "rank",
                    "source",
                    "page",
                    "distance",
                    "text",
                }

                missing = required_passage_fields - passage.keys()

                if missing:
                    raise RetrievalEvidenceError(
                        f"Passage {annotation_key} is missing fields: "
                        f"{sorted(missing)}"
                    )

                evidence.append({
                    "annotation_key": annotation_key,
                    "configuration": configuration,
                    "question_id": question_id,
                    "question": question_data.get("question", ""),
                    "rank": rank,
                    "relevance_grade": annotation["relevance_grade"],
                    "chunk_id": passage["chunk_id"],
                    "source": passage["source"],
                    "page": passage["page"],
                    "distance": passage["distance"],
                    "text": passage["text"],
                })

    unused_annotations = set(annotations) - matched_keys

    if unused_annotations:
        raise RetrievalEvidenceError(
            "Annotations without matching candidate passages: "
            f"{sorted(unused_annotations)[:10]}"
        )

    return evidence


if __name__ == "__main__":
    records = load_retrieval_evidence()

    print("=" * 60)
    print("RETRIEVAL EVIDENCE LOADED")
    print("=" * 60)
    print(f"Total joined passages: {len(records)}")
    print(
        "Grade 0:",
        sum(r["relevance_grade"] == 0 for r in records),
    )
    print(
        "Grade 1:",
        sum(r["relevance_grade"] == 1 for r in records),
    )
    print(
        "Grade 2:",
        sum(r["relevance_grade"] == 2 for r in records),
    )
