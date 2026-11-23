
from collections import Counter, defaultdict
from statistics import mean

from test.retrieval_evidence import load_retrieval_evidence


FAILURE_GRADES = {0, 1}


def average(values):
    return round(mean(values), 4) if values else 0.0


def group_records(records, field):
    groups = defaultdict(list)

    for record in records:
        groups[record[field]].append(record)

    return groups


def summarize_records(records):
    total = len(records)

    grade_counts = Counter(
        record["relevance_grade"] for record in records
    )

    failures = sum(
        record["relevance_grade"] in FAILURE_GRADES
        for record in records
    )

    distances = [
        record["distance"]
        for record in records
        if isinstance(record["distance"], (int, float))
    ]

    return {
        "total": total,
        "grade_0": grade_counts[0],
        "grade_1": grade_counts[1],
        "grade_2": grade_counts[2],
        "failure_count": failures,
        "failure_rate": round(failures / total, 4) if total else 0.0,
        "average_grade": average([
            record["relevance_grade"] for record in records
        ]),
        "average_distance": average(distances),
    }


def summarize_by(records, field):
    groups = group_records(records, field)

    return {
        str(key): summarize_records(group)
        for key, group in sorted(groups.items(), key=lambda item: str(item[0]))
    }


def analyze_retrieval_failures(records=None):
    """
    Investigate relevance failures using joined passage evidence.

    Grade 0 and grade 1 are grouped as failures for the summary,
    but their individual counts remain separate.
    """
    if records is None:
        records = load_retrieval_evidence()

    if not records:
        raise ValueError("No retrieval evidence was provided.")

    failure_records = [
        record for record in records
        if record["relevance_grade"] in FAILURE_GRADES
    ]

    # Repeated chunks across configurations.
    chunk_groups = defaultdict(list)

    for record in records:
        chunk_groups[record["chunk_id"]].append(record)

    repeated_chunks = []

    for chunk_id, occurrences in chunk_groups.items():
        configurations = sorted({
            record["configuration"] for record in occurrences
        })

        if len(configurations) > 1:
            repeated_chunks.append({
                "chunk_id": chunk_id,
                "configurations": configurations,
                "occurrence_count": len(occurrences),
                "grades": [
                    record["relevance_grade"] for record in occurrences
                ],
                "sources": sorted({
                    record["source"] for record in occurrences
                }),
            })

    repeated_chunks.sort(
        key=lambda item: (-item["occurrence_count"], item["chunk_id"])
    )

    # Preserve complete failure evidence for inspection.
    failure_details = []

    for record in failure_records:
        failure_details.append({
            **record,
            "failure_type": (
                "irrelevant"
                if record["relevance_grade"] == 0
                else "partially_relevant"
            ),
        })

    failure_details.sort(
        key=lambda item: (
            item["configuration"],
            item["question_id"],
            item["rank"],
        )
    )

    return {
        "overall": summarize_records(records),
        "failure_count": len(failure_records),
        "failure_rate": round(
            len(failure_records) / len(records), 4
        ),
        "by_configuration": summarize_by(
            records, "configuration"
        ),
        "by_question": summarize_by(
            records, "question_id"
        ),
        "by_source": summarize_by(
            records, "source"
        ),
        "by_rank": summarize_by(
            records, "rank"
        ),
        "failure_details": failure_details,
        "repeated_chunks": repeated_chunks,
        "notes": [
            "Grade 0 means irrelevant; grade 1 means partially relevant.",
            "Grade 2 means directly relevant.",
            "Retrieval distance is reported as stored; its interpretation "
            "depends on the vector database distance metric.",
            "Repeated chunk IDs indicate overlap across configurations, "
            "not necessarily a retrieval defect.",
            "The analysis covers the saved top-five candidate passages "
            "for each evaluated query, not the entire knowledge base.",
        ],
    }


if __name__ == "__main__":
    report = analyze_retrieval_failures()

    print("=" * 60)
    print("FAILURE INVESTIGATION")
    print("=" * 60)

    print("\nOVERALL")
    for key, value in report["overall"].items():
        print(f"{key}: {value}")

    print("\nBY CONFIGURATION")
    for name, summary in report["by_configuration"].items():
        print(name, summary)

    print("\nBY SOURCE")
    for name, summary in report["by_source"].items():
        print(name, summary)

    print("\nBY RANK")
    for name, summary in report["by_rank"].items():
        print(name, summary)

    print("\nRepeated chunks across configurations:")
    print(len(report["repeated_chunks"]))
