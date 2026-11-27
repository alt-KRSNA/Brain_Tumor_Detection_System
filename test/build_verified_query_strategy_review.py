
from pathlib import Path
import re

import pandas as pd

from genai.hybrid_retriever import HybridRetriever


ROOT = Path(__file__).resolve().parent.parent

REVIEW_FILE = (
    ROOT / "test/reports/query_strategy_experiment/"
    "query_strategy_review.csv"
)

DENSE_FILE = (
    ROOT / "test/reports/retrieval_baseline/"
    "baseline_relevance_annotations.csv"
)

HYBRID_FILE = (
    ROOT / "test/reports/hybrid_retrieval/"
    "hybrid_relevance_annotations.csv"
)

OUTPUT_FILE = (
    ROOT / "test/reports/query_strategy_experiment/"
    "query_strategy_review_identity_verified.csv"
)


def normalize_text(value):
    """Normalize whitespace and case without removing meaningful text."""
    if pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip().lower()


def normalize_source(value):
    if pd.isna(value):
        return ""

    return Path(str(value)).name.strip().lower()


def normalize_page(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def load_annotations(path, text_column, label_column):
    if not path.exists():
        print(f"Annotation file missing: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    required = {
        "question_id",
        "source",
        "page",
        text_column,
        label_column,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path.name}: {sorted(missing)}"
        )

    df = df.copy()

    df["identity_source"] = df["source"].apply(normalize_source)
    df["identity_page"] = df["page"].apply(normalize_page)
    df["identity_text"] = df[text_column].apply(normalize_text)

    return df


def main():
    print("=" * 90)
    print("BUILDING IDENTITY-VERIFIED QUERY STRATEGY REVIEW")
    print("=" * 90)

    review = pd.read_csv(REVIEW_FILE)

    retriever = HybridRetriever()
    collection = retriever.collection

    candidate_ids = (
        review["passage_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    chroma_data = collection.get(
        ids=candidate_ids,
        include=["documents", "metadatas"],
    )

    candidate_lookup = {}

    for passage_id, document, metadata in zip(
        chroma_data["ids"],
        chroma_data["documents"],
        chroma_data["metadatas"],
    ):
        metadata = metadata or {}

        candidate_lookup[str(passage_id)] = {
            "source": metadata.get("source", ""),
            "page": metadata.get("page", ""),
            "passage_text": document or "",
        }

    dense = load_annotations(
        DENSE_FILE,
        text_column="text",
        label_column="relevance_grade",
    )

    hybrid = load_annotations(
        HYBRID_FILE,
        text_column="passage_text",
        label_column="relevance_grade",
    )

    annotations = pd.concat(
        [dense, hybrid],
        ignore_index=True,
    )

    annotations["identity_question"] = (
        annotations["question_id"].astype(str)
    )

    annotations["identity_label"] = pd.to_numeric(
        annotations["relevance_grade"],
        errors="coerce",
    )

    # Only annotations with a complete identity and an actual grade
    # are eligible for reuse.
    annotations = annotations[
        (annotations["identity_source"] != "")
        & (annotations["identity_page"] != "")
        & (annotations["identity_text"] != "")
        & (annotations["identity_label"].notna())
    ]

    verified_grades = []
    identity_statuses = []
    matched_sources = []

    for _, row in review.iterrows():
        passage_id = str(row["passage_id"])
        question_id = str(row["question_id"])

        candidate = candidate_lookup.get(passage_id)

        if candidate is None:
            verified_grades.append(pd.NA)
            identity_statuses.append("MISSING_FROM_CHROMA")
            matched_sources.append("")
            continue

        source = normalize_source(candidate["source"])
        page = normalize_page(candidate["page"])
        text = normalize_text(candidate["passage_text"])

        if not source or not page or not text:
            verified_grades.append(pd.NA)
            identity_statuses.append("INCOMPLETE_CHROMA_METADATA")
            matched_sources.append("")
            continue

        matches = annotations[
            (annotations["identity_question"] == question_id)
            & (annotations["identity_source"] == source)
            & (annotations["identity_page"] == page)
            & (annotations["identity_text"] == text)
        ]

        grades = sorted(
            matches["identity_label"].dropna().unique().tolist()
        )

        if len(grades) == 1:
            verified_grades.append(int(grades[0]))
            identity_statuses.append("EXACT_MATCH_GRADE_REUSED")

        elif len(grades) > 1:
            verified_grades.append(pd.NA)
            identity_statuses.append("EXACT_MATCH_GRADE_CONFLICT")

        else:
            verified_grades.append(pd.NA)
            identity_statuses.append("NEW_HUMAN_REVIEW_REQUIRED")

        matched_sources.append(
            "; ".join(sorted(matches["source"].dropna().astype(str).unique()))
        )

    output = review.copy()

    output["source"] = output["passage_id"].astype(str).map(
        lambda pid: candidate_lookup.get(pid, {}).get("source", "")
    )

    output["page"] = output["passage_id"].astype(str).map(
        lambda pid: candidate_lookup.get(pid, {}).get("page", "")
    )

    output["passage_text"] = output["passage_id"].astype(str).map(
        lambda pid: candidate_lookup.get(pid, {}).get("passage_text", "")
    )

    output["verified_relevance_grade"] = verified_grades
    output["identity_status"] = identity_statuses
    output["matched_annotation_sources"] = matched_sources

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 90)
    print("IDENTITY VERIFICATION SUMMARY")
    print("=" * 90)

    print(f"Review rows: {len(output)}")
    print(f"Unique passage IDs: {len(candidate_ids)}")
    print(f"Chroma IDs found: {len(candidate_lookup)}")

    print("\nIdentity status:")
    print(output["identity_status"].value_counts().to_string())

    print("\nVerified grade distribution:")
    print(
        output["verified_relevance_grade"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("\nOriginal annotation files were not modified.")


if __name__ == "__main__":
    main()
