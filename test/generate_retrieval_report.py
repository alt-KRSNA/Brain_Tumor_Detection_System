import csv
from pathlib import Path

import matplotlib.pyplot as plt

from test.retrieval_experiment_data import (
    CONFIGURATION_RESULTS,
    ANNOTATION_STATUS,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "test" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_rankings(metric):
    return sorted(
        CONFIGURATION_RESULTS.items(),
        key=lambda item: item[1][metric],
        reverse=True,
    )


def export_comparison_csv():
    output_file = OUTPUT_DIR / "retrieval_comparison.csv"

    metrics = [
        "mrr",
        "precision_at_3",
        "recall_at_3",
        "ndcg_at_3",
        "avg_relevance_at_3",
        "precision_at_5",
        "recall_at_5",
        "ndcg_at_5",
        "avg_relevance_at_5",
    ]

    headers = [
        "configuration",
        "chunk_size",
        "overlap",
        *metrics,
    ]

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=headers,
        )

        writer.writeheader()

        for config, values in CONFIGURATION_RESULTS.items():

            row = {
                "configuration": config,
                "chunk_size": values["chunk_size"],
                "overlap": values["overlap"],
            }

            row.update({
                metric: values[metric]
                for metric in metrics
            })

            writer.writerow(row)

    return output_file


def generate_markdown_report():

    report_file = OUTPUT_DIR / "retrieval_experiment_report.md"

    ranking_ndcg = get_rankings("ndcg_at_5")
    ranking_precision = get_rankings("precision_at_5")

    lines = [
        "# Retrieval Chunking Experiment Report",
        "",
        "> Relevance labels are provisional model-assisted judgments.",
        "> This report should not be treated as final validated research.",
        "",
        "## Experimental configurations",
        "",
        "| Configuration | Chunk size | Overlap |",
        "|---|---:|---:|",
    ]

    for config, values in CONFIGURATION_RESULTS.items():
        lines.append(
            f"| {config} | {values['chunk_size']} | "
            f"{values['overlap']} |"
        )

    lines.extend([
        "",
        "## Aggregate metric comparison",
        "",
        "| Config | MRR | P@3 | R@3 | nDCG@3 | "
        "P@5 | R@5 | nDCG@5 | AvgRel@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for config, values in CONFIGURATION_RESULTS.items():
        lines.append(
            f"| {config} | "
            f"{values['mrr']:.4f} | "
            f"{values['precision_at_3']:.4f} | "
            f"{values['recall_at_3']:.4f} | "
            f"{values['ndcg_at_3']:.4f} | "
            f"{values['precision_at_5']:.4f} | "
            f"{values['recall_at_5']:.4f} | "
            f"{values['ndcg_at_5']:.4f} | "
            f"{values['avg_relevance_at_5']:.4f} |"
        )

    lines.extend([
        "",
        "## Ranking by nDCG@5",
        "",
    ])

    for rank, (config, values) in enumerate(
        ranking_ndcg,
        start=1,
    ):
        lines.append(
            f"{rank}. Config {config}: "
            f"{values['ndcg_at_5']:.4f}"
        )

    lines.extend([
        "",
        "## Ranking by Precision@5",
        "",
    ])

    for rank, (config, values) in enumerate(
        ranking_precision,
        start=1,
    ):
        lines.append(
            f"{rank}. Config {config}: "
            f"{values['precision_at_5']:.4f}"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Config A achieved the highest nDCG@5.",
        "- Config B achieved the highest Precision@5.",
        "- Config C had the lowest nDCG@5.",
        "- Recall@5 equals 1.0 for all configurations.",
        "",
        "These results are based on a small query set and provisional",
        "relevance judgments. Validate the annotations and expand the",
        "evaluation queries before making a final selection.",
        "",
    ])

    report_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return report_file


def generate_metric_chart(metric, title, filename):

    configurations = list(CONFIGURATION_RESULTS.keys())

    values = [
        CONFIGURATION_RESULTS[config][metric]
        for config in configurations
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(configurations, values)

    plt.title(title)
    plt.xlabel("Chunking configuration")
    plt.ylabel(metric)
    plt.ylim(0, 1.05)

    plt.tight_layout()

    output_file = OUTPUT_DIR / filename

    plt.savefig(output_file, dpi=150)
    plt.close()

    return output_file


def main():

    csv_file = export_comparison_csv()
    report_file = generate_markdown_report()

    ndcg_chart = generate_metric_chart(
        "ndcg_at_5",
        "Retrieval Ranking Quality (nDCG@5)",
        "ndcg_at_5.png",
    )

    precision_chart = generate_metric_chart(
        "precision_at_5",
        "Relevant Passages (Precision@5)",
        "precision_at_5.png",
    )

    print("=" * 60)
    print("RETRIEVAL EXPERIMENT REPORT GENERATED")
    print("=" * 60)

    print(f"Annotation status : {ANNOTATION_STATUS}")
    print(f"CSV comparison   : {csv_file}")
    print(f"Markdown report  : {report_file}")
    print(f"nDCG chart      : {ndcg_chart}")
    print(f"Precision chart  : {precision_chart}")


if __name__ == "__main__":
    main()