
"""Generate Markdown and CSV reports for retrieval failure analysis."""

import csv
from pathlib import Path

from test.failure_analysis import analyze_file


OUTPUT_DIR = Path(__file__).parent / "reports"


def export_failure_records_csv(results, output_path):
    """Export grade-0 and grade-1 passage records."""

    fields = [
        "annotation_key",
        "configuration",
        "question_id",
        "rank",
        "relevance_grade",
        "grade_label",
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results["failure_records"])

    return output_path


def export_summary_csv(results, output_path):
    """Export configuration, query, and rank summary rows."""

    rows = []

    for config, metrics in results[
        "configuration_analysis"
    ].items():
        rows.append({
            "group_type": "configuration",
            "group_name": config,
            **metrics,
        })

    for query, metrics in results["query_analysis"].items():
        rows.append({
            "group_type": "query",
            "group_name": query,
            **metrics,
        })

    for rank, metrics in results["rank_analysis"].items():
        rows.append({
            "group_type": "rank",
            "group_name": str(rank),
            **metrics,
        })

    fields = [
        "group_type",
        "group_name",
        "total",
        "grade_0",
        "grade_1",
        "grade_2",
        "failure_count",
        "failure_rate",
        "irrelevant_rate",
        "partial_rate",
        "relevant_rate",
        "average_grade",
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def generate_markdown_report(results, output_path):
    """Generate a Markdown report from failure analysis."""

    overall = results["overall"]
    configs = results["configuration_analysis"]
    queries = results["query_analysis"]
    ranks = results["rank_analysis"]

    lines = [
        "# Retrieval Failure Analysis Report",
        "",
        "## Evaluation Overview",
        "",
        f"- Total passages evaluated: {overall['total_passages']}",
        f"- Queries: {results['query_count']}",
        f"- Configurations: {results['configuration_count']}",
        f"- Grade 0 (irrelevant): {overall['grade_0_count']}",
        f"- Grade 1 (partial): {overall['grade_1_count']}",
        f"- Grade 2 (relevant): {overall['grade_2_count']}",
        f"- Total failures: {overall['failure_count']}",
        f"- Failure rate: {overall['failure_rate']:.2%}",
        "",
        "> Failure is defined as a retrieved passage with relevance grade 0 or 1.",
        "> Annotation labels are provisional and model-assisted.",
        "",
        "## Configuration Comparison",
        "",
        "| Config | Total | Grade 0 | Grade 1 | Grade 2 | Failures | Failure Rate | Avg Grade |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for config, metrics in configs.items():
        lines.append(
            f"| {config} | "
            f"{metrics['total']} | "
            f"{metrics['grade_0']} | "
            f"{metrics['grade_1']} | "
            f"{metrics['grade_2']} | "
            f"{metrics['failure_count']} | "
            f"{metrics['failure_rate']:.2%} | "
            f"{metrics['average_grade']:.2f} |"
        )

    lines.extend([
        "",
        f"- Highest relevant rate: **{results['best_configuration']}**",
        f"- Highest failure rate: **{results['worst_configuration']}**",
        "",
        "## Query Analysis",
        "",
        "| Query | Total | Grade 0 | Grade 1 | Grade 2 | Failure Rate | Avg Grade |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])

    for query, metrics in queries.items():
        lines.append(
            f"| {query} | "
            f"{metrics['total']} | "
            f"{metrics['grade_0']} | "
            f"{metrics['grade_1']} | "
            f"{metrics['grade_2']} | "
            f"{metrics['failure_rate']:.2%} | "
            f"{metrics['average_grade']:.2f} |"
        )

    lines.extend([
        "",
        f"- Highest failure query: **{results['worst_query']}**",
        "",
        "## Rank Analysis",
        "",
        "| Rank | Total | Grade 0 | Grade 1 | Grade 2 | Failure Rate | Avg Grade |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])

    for rank, metrics in ranks.items():
        lines.append(
            f"| {rank} | "
            f"{metrics['total']} | "
            f"{metrics['grade_0']} | "
            f"{metrics['grade_1']} | "
            f"{metrics['grade_2']} | "
            f"{metrics['failure_rate']:.2%} | "
            f"{metrics['average_grade']:.2f} |"
        )

    lines.extend([
        "",
        "## Failure Records",
        "",
        "| Config | Query | Rank | Grade | Label |",
        "|---|---|---:|---:|---|",
    ])

    for failure in results["failure_records"]:
        lines.append(
            f"| {failure['configuration']} | "
            f"{failure['question_id']} | "
            f"{failure['rank']} | "
            f"{failure['relevance_grade']} | "
            f"{failure['grade_label']} |"
        )

    lines.extend([
        "",
        "## Interpretation and Limitations",
        "",
        "- Grade 0 indicates an irrelevant retrieved passage.",
        "- Grade 1 indicates partial relevance.",
        "- Grade 2 indicates direct relevance.",
        "- A failure rate of grades 0 + 1 measures retrieval quality "
        "using the current annotation rubric.",
        "- This report analyzes relevance grades only. It does not "
        "inspect passage text or establish why a retrieval failed.",
        "- Annotation labels are provisional, model-assisted judgments "
        "and require human validation.",
        "- Results are based on the current evaluation queries and "
        "retrieved passage pool, not the entire knowledge base.",
        "",
    ])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines), encoding="utf-8"
    )

    return output_path


def generate_reports(annotation_path=None, output_dir=OUTPUT_DIR):
    """Run failure analysis and generate all reports."""

    if annotation_path is None:
        results = analyze_file()
    else:
        results = analyze_file(annotation_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = generate_markdown_report(
        results,
        output_dir / "failure_analysis_report.md",
    )

    failure_csv_path = export_failure_records_csv(
        results,
        output_dir / "failure_analysis.csv",
    )

    summary_csv_path = export_summary_csv(
        results,
        output_dir / "failure_summary.csv",
    )

    return {
        "results": results,
        "markdown": markdown_path,
        "failure_csv": failure_csv_path,
        "summary_csv": summary_csv_path,
    }


if __name__ == "__main__":
    generated = generate_reports()

    print("=" * 60)
    print("FAILURE ANALYSIS REPORT GENERATED")
    print("=" * 60)
    print(f"Markdown : {generated['markdown']}")
    print(f"Failures: {generated['failure_csv']}")
    print(f"Summary : {generated['summary_csv']}")
