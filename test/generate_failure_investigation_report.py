
import csv
from pathlib import Path

from test.failure_investigator import analyze_retrieval_failures


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "test" / "reports"

MARKDOWN_FILE = REPORT_DIR / "failure_investigation_report.md"
CSV_FILE = REPORT_DIR / "failure_evidence.csv"


def format_summary_table(title, summaries):
    lines = [
        f"### {title}",
        "",
        "| Group | Total | Grade 0 | Grade 1 | Grade 2 | "
        "Failure rate | Average grade | Average distance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, data in summaries.items():
        lines.append(
            f"| {name} | {data['total']} | {data['grade_0']} | "
            f"{data['grade_1']} | {data['grade_2']} | "
            f"{data['failure_rate']:.2%} | "
            f"{data['average_grade']:.4f} | "
            f"{data['average_distance']:.6f} |"
        )

    lines.append("")
    return lines


def build_markdown(report):
    overall = report["overall"]

    lines = [
        "# Retrieval Failure Investigation",
        "",
        "## 1. Overall results",
        "",
        f"- Total evaluated passages: **{overall['total']}**",
        f"- Grade 0 (irrelevant): **{overall['grade_0']}**",
        f"- Grade 1 (partially relevant): **{overall['grade_1']}**",
        f"- Grade 2 (directly relevant): **{overall['grade_2']}**",
        f"- Grade 0 + Grade 1: **{report['failure_count']}**",
        f"- Combined failure rate: **{report['failure_rate']:.2%}**",
        f"- Average relevance grade: **{overall['average_grade']:.4f}**",
        f"- Average retrieval distance: **{overall['average_distance']:.6f}**",
        "",
    ]

    lines.extend(
        format_summary_table(
            "2. Results by configuration",
            report["by_configuration"],
        )
    )

    lines.extend(
        format_summary_table(
            "3. Results by question",
            report["by_question"],
        )
    )

    lines.extend(
        format_summary_table(
            "4. Results by source PDF",
            report["by_source"],
        )
    )

    lines.extend(
        format_summary_table(
            "5. Results by rank",
            report["by_rank"],
        )
    )

    lines.extend([
        "## 6. Repeated chunks across configurations",
        "",
        "| Chunk ID | Configurations | Occurrences | Grades | Sources |",
        "|---|---|---:|---|---|",
    ])

    if report["repeated_chunks"]:
        for item in report["repeated_chunks"]:
            lines.append(
                f"| {item['chunk_id']} | "
                f"{', '.join(item['configurations'])} | "
                f"{item['occurrence_count']} | "
                f"{', '.join(map(str, item['grades']))} | "
                f"{', '.join(item['sources'])} |"
            )
    else:
        lines.append("| None | — | 0 | — | — |")

    lines.extend([
        "",
        "## 7. Failure evidence",
        "",
        "The following records are the actual retrieved passages graded "
        "0 or 1. Inspect their text and source context before deciding "
        "what retrieval change is justified.",
        "",
    ])

    for record in report["failure_details"]:
        lines.extend([
            f"### {record['annotation_key']}",
            "",
            f"- Failure type: **{record['failure_type']}**",
            f"- Relevance grade: **{record['relevance_grade']}**",
            f"- Question: {record['question']}",
            f"- Chunk ID: `{record['chunk_id']}`",
            f"- Source: `{record['source']}`",
            f"- Page: {record['page']}",
            f"- Rank: {record['rank']}",
            f"- Distance: {record['distance']}",
            "",
            "**Retrieved passage:**",
            "",
            "```text",
            record["text"].replace("```", "'''"),
            "```",
            "",
        ])

    lines.extend([
        "## 8. Interpretation notes",
        "",
    ])

    for note in report["notes"]:
        lines.append(f"- {note}")

    lines.extend([
        "",
        "## 9. Scope and limitations",
        "",
        "- This investigation uses the saved candidate-passage pool and "
        "its corresponding annotations.",
        "- The results do not establish that a source PDF is generally "
        "poor or that a particular chunking configuration is universally "
        "better.",
        "- Distance values should not be interpreted as relevance scores "
        "without confirming the configured distance metric.",
        "- Grade 1 passages should be reviewed separately from grade 0 "
        "passages; partial relevance is not the same as irrelevance.",
        "",
    ])

    return "\n".join(lines)


def write_failure_csv(records, output_path=CSV_FILE):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "annotation_key",
        "failure_type",
        "configuration",
        "question_id",
        "question",
        "rank",
        "relevance_grade",
        "chunk_id",
        "source",
        "page",
        "distance",
        "text",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow({
                field: record.get(field, "")
                for field in fieldnames
            })


def generate_report():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = analyze_retrieval_failures()

    markdown = build_markdown(report)
    MARKDOWN_FILE.write_text(markdown, encoding="utf-8")

    write_failure_csv(report["failure_details"])

    print("=" * 60)
    print("FAILURE INVESTIGATION REPORT GENERATED")
    print("=" * 60)
    print(f"Markdown: {MARKDOWN_FILE}")
    print(f"CSV     : {CSV_FILE}")
    print(f"Passages: {report['overall']['total']}")
    print(f"Failures: {report['failure_count']}")
    print(f"Failure rate: {report['failure_rate']:.2%}")


if __name__ == "__main__":
    generate_report()
