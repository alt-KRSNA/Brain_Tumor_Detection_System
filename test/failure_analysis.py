
"""Analyze retrieval failures from passage relevance annotations."""

from collections import Counter, defaultdict
from pathlib import Path
import json


DEFAULT_ANNOTATION_PATH = (
    Path(__file__).parent / "passage_annotations.json"
)

GRADE_LABELS = {
    0: "irrelevant",
    1: "partially_relevant",
    2: "directly_relevant",
}


def load_annotations(path=DEFAULT_ANNOTATION_PATH):
    """Load and validate the annotation JSON."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Annotation root must be a dictionary.")

    validated = {}

    for key, record in data.items():
        if not isinstance(record, dict):
            raise ValueError(f"Invalid record for key: {key}")

        required = {
            "configuration",
            "question_id",
            "rank",
            "relevance_grade",
        }

        missing = required - record.keys()

        if missing:
            raise ValueError(
                f"Missing fields in {key}: {sorted(missing)}"
            )

        grade = record["relevance_grade"]

        if grade not in (0, 1, 2):
            raise ValueError(
                f"Invalid relevance grade {grade} in {key}"
            )

        validated[key] = {
            "configuration": str(record["configuration"]),
            "question_id": str(record["question_id"]),
            "rank": int(record["rank"]),
            "relevance_grade": int(grade),
        }

    return validated


def analyze_annotations(annotations):
    """Analyze passage-level retrieval relevance and failures."""

    if not annotations:
        raise ValueError("No annotation records provided.")

    records = list(annotations.values())

    configurations = sorted({
        record["configuration"] for record in records
    })

    queries = sorted({
        record["question_id"] for record in records
    })

    total = len(records)

    grade_counts = Counter(
        record["relevance_grade"] for record in records
    )

    failure_count = grade_counts[0] + grade_counts[1]

    overall = {
        "total_passages": total,
        "grade_0_count": grade_counts[0],
        "grade_1_count": grade_counts[1],
        "grade_2_count": grade_counts[2],
        "irrelevant_rate": grade_counts[0] / total,
        "partial_rate": grade_counts[1] / total,
        "relevant_rate": grade_counts[2] / total,
        "failure_count": failure_count,
        "failure_rate": failure_count / total,
    }

    config_groups = defaultdict(list)
    query_groups = defaultdict(list)
    config_query_groups = defaultdict(list)
    rank_groups = defaultdict(list)

    for record in records:
        config = record["configuration"]
        query = record["question_id"]
        rank = record["rank"]
        grade = record["relevance_grade"]

        config_groups[config].append(grade)
        query_groups[query].append(grade)
        config_query_groups[(config, query)].append(grade)
        rank_groups[rank].append(grade)

    def summarize_grades(grades):
        """Calculate summary metrics for a group of relevance grades."""

        count = len(grades)
        grade_counter = Counter(grades)
        failures = grade_counter[0] + grade_counter[1]

        return {
            "total": count,
            "grade_0": grade_counter[0],
            "grade_1": grade_counter[1],
            "grade_2": grade_counter[2],
            "failure_count": failures,
            "failure_rate": failures / count if count else 0.0,
            "irrelevant_rate": (
                grade_counter[0] / count if count else 0.0
            ),
            "partial_rate": (
                grade_counter[1] / count if count else 0.0
            ),
            "relevant_rate": (
                grade_counter[2] / count if count else 0.0
            ),
            "average_grade": (
                sum(grades) / count if count else 0.0
            ),
        }

    configuration_analysis = {
        config: summarize_grades(grades)
        for config, grades in sorted(config_groups.items())
    }

    query_analysis = {
        query: summarize_grades(grades)
        for query, grades in sorted(query_groups.items())
    }

    configuration_query_analysis = {
        f"{config}|{query}": {
            "configuration": config,
            "question_id": query,
            **summarize_grades(grades),
        }
        for (config, query), grades
        in sorted(config_query_groups.items())
    }

    rank_analysis = {
        rank: summarize_grades(grades)
        for rank, grades in sorted(rank_groups.items())
    }

    failure_records = []

    for key, record in annotations.items():
        grade = record["relevance_grade"]

        if grade < 2:
            failure_records.append({
                "annotation_key": key,
                "configuration": record["configuration"],
                "question_id": record["question_id"],
                "rank": record["rank"],
                "relevance_grade": grade,
                "grade_label": GRADE_LABELS[grade],
            })

    failure_records.sort(
        key=lambda item: (
            item["configuration"],
            item["question_id"],
            item["rank"],
        )
    )

    # Highest failure rate; break ties using lowest average relevance.
    worst_configuration = max(
        configuration_analysis,
        key=lambda config: (
            configuration_analysis[config]["failure_rate"],
            -configuration_analysis[config]["average_grade"],
        ),
    )

    # Highest directly-relevant rate; break ties using highest average grade.
    best_configuration = max(
        configuration_analysis,
        key=lambda config: (
            configuration_analysis[config]["relevant_rate"],
            configuration_analysis[config]["average_grade"],
        ),
    )

    # Highest failure rate; break ties using lowest average relevance.
    worst_query = max(
        query_analysis,
        key=lambda query: (
            query_analysis[query]["failure_rate"],
            -query_analysis[query]["average_grade"],
        ),
    )

    return {
        "overall": overall,
        "configuration_analysis": configuration_analysis,
        "query_analysis": query_analysis,
        "configuration_query_analysis": (
            configuration_query_analysis
        ),
        "rank_analysis": rank_analysis,
        "failure_records": failure_records,
        "worst_configuration": worst_configuration,
        "best_configuration": best_configuration,
        "worst_query": worst_query,
        "query_count": len(queries),
        "configuration_count": len(configurations),
        "configurations": configurations,
        "queries": queries,
    }


def analyze_file(path=DEFAULT_ANNOTATION_PATH):
    """Load annotation JSON and return failure analysis."""
    annotations = load_annotations(path)
    return analyze_annotations(annotations)


if __name__ == "__main__":
    results = analyze_file()

    print("=" * 60)
    print("RETRIEVAL FAILURE ANALYSIS")
    print("=" * 60)

    overall = results["overall"]

    print(f"Total passages        : {overall['total_passages']}")
    print(f"Irrelevant (Grade 0) : {overall['grade_0_count']}")
    print(f"Partial (Grade 1)     : {overall['grade_1_count']}")
    print(f"Relevant (Grade 2)    : {overall['grade_2_count']}")
    print(f"Failure rate          : {overall['failure_rate']:.2%}")
    print(f"Worst configuration  : {results['worst_configuration']}")
    print(f"Best configuration   : {results['best_configuration']}")
    print(f"Worst query          : {results['worst_query']}")

    print("\nCONFIGURATION BREAKDOWN")

    for config, metrics in results[
        "configuration_analysis"
    ].items():
        print(
            f"{config}: "
            f"failures={metrics['failure_count']}/{metrics['total']} "
            f"({metrics['failure_rate']:.2%}), "
            f"avg_grade={metrics['average_grade']:.2f}"
        )

    print("\nFAILURES BY RANK")

    for rank, metrics in results["rank_analysis"].items():
        print(
            f"Rank {rank}: "
            f"failures={metrics['failure_count']}/{metrics['total']} "
            f"({metrics['failure_rate']:.2%})"
        )

    print("\nLOW-RELEVANCE PASSAGES")

    for item in results["failure_records"]:
        print(
            f"{item['annotation_key']} | "
            f"Grade {item['relevance_grade']} | "
            f"{item['grade_label']}"
        )
