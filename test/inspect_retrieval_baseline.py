# test/inspect_retrieval_baseline.py

from pathlib import Path
import csv

import chromadb
from sentence_transformers import SentenceTransformer

from test.experiment_retrieval import (
    CONFIGURATIONS,
    QUESTIONS,
    MODEL_NAME,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = PROJECT_ROOT / "vectorstore" / "experiments"

OUTPUT_DIR = PROJECT_ROOT / "test" / "reports" / "retrieval_baseline"

TOP_K = 5


def get_collection_name(config):
    """
    Return the experiment collection name for a configuration.

    A - Baseline -> experiment_chunking_a
    B - Medium    -> experiment_chunking_b
    C - Small     -> experiment_chunking_c
    """
    config_code = config["name"].split()[0].lower()

    return f"experiment_chunking_{config_code}"


def retrieve_top_passages(collection, model, question, top_k=TOP_K):
    """
    Retrieve the top-K passages for a single evaluation question.
    """

    query_embedding = model.encode(
        [question["question"]],
        convert_to_numpy=True,
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    passages = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for rank, (chunk_id, text, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        passages.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", "unknown"),
                "distance": float(distance),
                "text": text,
            }
        )

    return passages


def summarize_question_records(records):
    """
    Calculate source-level summary statistics.

    IMPORTANT:
    These statistics do NOT establish passage relevance.
    Source matching is only a proxy for retrieval quality.
    """

    if not records:
        return {
            "total_questions": 0,
            "source_hit_at_5": 0.0,
            "top1_expected_source_rate": 0.0,
            "average_expected_source_count_at_5": 0.0,
            "average_top1_distance": 0.0,
        }

    total_questions = len(records)

    source_hits = 0
    top1_expected_source_hits = 0
    expected_source_count_total = 0
    top1_distance_total = 0.0

    for record in records:
        expected_sources = set(record["expected_sources"])
        passages = record["passages"]

        retrieved_sources = [
            passage["source"]
            for passage in passages
        ]

        # Did any retrieved passage come from an expected source?
        if any(
            source in expected_sources
            for source in retrieved_sources
        ):
            source_hits += 1

        # Did rank 1 come from an expected source?
        if passages and passages[0]["source"] in expected_sources:
            top1_expected_source_hits += 1

        expected_source_count_total += sum(
            source in expected_sources
            for source in retrieved_sources
        )

        if passages:
            top1_distance_total += passages[0]["distance"]

    return {
        "total_questions": total_questions,
        "source_hit_at_5": source_hits / total_questions,
        "top1_expected_source_rate": (
            top1_expected_source_hits / total_questions
        ),
        "average_expected_source_count_at_5": (
            expected_source_count_total / total_questions
        ),
        "average_top1_distance": (
            top1_distance_total / total_questions
        ),
    }


def build_question_summary(config_name, record):
    """
    Create a summary row for one configuration and one question.
    """

    expected_sources = set(record["expected_sources"])
    passages = record["passages"]

    matching_passages = [
        passage
        for passage in passages
        if passage["source"] in expected_sources
    ]

    first_matching_rank = (
        matching_passages[0]["rank"]
        if matching_passages
        else None
    )

    top1_expected_source = (
        passages[0]["source"] in expected_sources
        if passages
        else False
    )

    top1_distance = (
        passages[0]["distance"]
        if passages
        else None
    )

    return {
        "configuration": config_name,
        "question": record["question"],
        "expected_sources": ";".join(
            sorted(expected_sources)
        ),
        "top1_source": (
            passages[0]["source"]
            if passages
            else ""
        ),
        "top1_expected_source": top1_expected_source,
        "first_expected_source_rank": first_matching_rank,
        "expected_source_passages_at_5": len(matching_passages),
        "top1_distance": top1_distance,
    }


def write_passage_csv(records, output_path):
    """
    Export every retrieved passage as an individual CSV row.
    """

    fieldnames = [
        "configuration",
        "question_id",
        "question",
        "expected_sources",
        "rank",
        "chunk_id",
        "source",
        "page",
        "distance",
        "text",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            for passage in record["passages"]:
                writer.writerow(
                    {
                        "configuration": record["configuration"],
                        "question_id": record["question_id"],
                        "question": record["question"],
                        "expected_sources": ";".join(
                            record["expected_sources"]
                        ),
                        "rank": passage["rank"],
                        "chunk_id": passage["chunk_id"],
                        "source": passage["source"],
                        "page": passage["page"],
                        "distance": passage["distance"],
                        "text": passage["text"],
                    }
                )


def write_summary_csv(summary_rows, output_path):
    """
    Export configuration-question source-level results.
    """

    fieldnames = [
        "configuration",
        "question",
        "expected_sources",
        "top1_source",
        "top1_expected_source",
        "first_expected_source_rank",
        "expected_source_passages_at_5",
        "top1_distance",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary_rows)


def write_markdown_report(records, summary_rows, output_path):
    """
    Generate a readable report showing top-K passages
    for every question and configuration.
    """

    lines = [
        "# Retrieval Baseline Inspection",
        "",
        f"Top-K: {TOP_K}",
        "",
        (
            "This report records retrieved passages and source-level "
            "matching. Source matching is not equivalent to passage "
            "relevance or answer correctness."
        ),
        "",
    ]

    for config in CONFIGURATIONS:
        config_name = config["name"]

        config_records = [
            record
            for record in records
            if record["configuration"] == config_name
        ]

        config_summaries = [
            row
            for row in summary_rows
            if row["configuration"] == config_name
        ]

        metrics = summarize_question_records(
            config_records
        )

        lines.extend(
            [
                "---",
                "",
                f"## {config_name}",
                "",
                f"- Chunk size: {config['size']}",
                f"- Overlap: {config['overlap']}",
                f"- Questions: {metrics['total_questions']}",
                (
                    "- Source Hit@5: "
                    f"{metrics['source_hit_at_5']:.4f}"
                ),
                (
                    "- Top-1 expected-source rate: "
                    f"{metrics['top1_expected_source_rate']:.4f}"
                ),
                (
                    "- Average expected-source passages in top 5: "
                    f"{metrics['average_expected_source_count_at_5']:.2f}"
                ),
                (
                    "- Average top-1 cosine distance: "
                    f"{metrics['average_top1_distance']:.4f}"
                ),
                "",
            ]
        )

        for record, summary in zip(
            config_records,
            config_summaries,
        ):
            lines.extend(
                [
                    f"### {record['question_id']}: {record['question']}",
                    "",
                    f"Expected sources: {', '.join(record['expected_sources'])}",
                    "",
                    (
                        "Top-1 expected source: "
                        f"{summary['top1_expected_source']}"
                    ),
                    (
                        "First expected-source rank: "
                        f"{summary['first_expected_source_rank']}"
                    ),
                    "",
                ]
            )

            for passage in record["passages"]:
                excerpt = passage["text"].replace(
                    "\n",
                    " ",
                ).strip()

                lines.extend(
                    [
                        (
                            f"#### Rank {passage['rank']} — "
                            f"{passage['source']} | Page {passage['page']}"
                        ),
                        "",
                        f"Chunk ID: `{passage['chunk_id']}`",
                        "",
                        f"Distance: `{passage['distance']:.4f}`",
                        "",
                        excerpt,
                        "",
                    ]
                )

    lines.extend(
        [
            "---",
            "",
            "## Interpretation limitations",
            "",
            "- Expected-source matching is not passage-level relevance.",
            "- A passage from an expected PDF may still be irrelevant.",
            "- A passage from a different PDF may still contain a valid answer.",
            "- Distance values are meaningful for relative comparisons "
            "within the same embedding/model/index setup.",
            "- Human relevance grading is required for stronger evaluation.",
            "",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    print("=" * 80)
    print("RETRIEVAL BASELINE INSPECTION")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(EXPERIMENT_DIR)
    )

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    all_records = []
    summary_rows = []

    for config in CONFIGURATIONS:
        config_name = config["name"]
        collection_name = get_collection_name(config)

        print("\n" + "=" * 80)
        print(config_name)
        print("Collection:", collection_name)
        print("=" * 80)

        try:
            collection = client.get_collection(
                name=collection_name
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load collection '{collection_name}'. "
                "Run the existing chunking experiment first."
            ) from exc

        print("Collection chunks:", collection.count())

        for question_index, question in enumerate(
            QUESTIONS,
            start=1,
        ):
            print(
                f"\nQuestion {question_index}/{len(QUESTIONS)}: "
                f"{question['question']}"
            )

            passages = retrieve_top_passages(
                collection=collection,
                model=model,
                question=question,
                top_k=TOP_K,
            )

            record = {
                "configuration": config_name,
                "question_id": f"Q{question_index:02d}",
                "question": question["question"],
                "expected_sources": question["expected_sources"],
                "passages": passages,
            }

            all_records.append(record)

            summary_rows.append(
                build_question_summary(
                    config_name,
                    record,
                )
            )

            for passage in passages:
                print(
                    f"  Rank {passage['rank']} | "
                    f"{passage['source']} | "
                    f"Page {passage['page']} | "
                    f"Distance {passage['distance']:.4f}"
                )

    passage_csv = OUTPUT_DIR / "baseline_passages.csv"
    summary_csv = OUTPUT_DIR / "baseline_summary.csv"
    markdown_report = OUTPUT_DIR / "baseline_inspection.md"

    write_passage_csv(
        all_records,
        passage_csv,
    )

    write_summary_csv(
        summary_rows,
        summary_csv,
    )

    write_markdown_report(
        all_records,
        summary_rows,
        markdown_report,
    )

    print("\n" + "=" * 80)
    print("BASELINE INSPECTION COMPLETED")
    print("=" * 80)

    print("\nFiles generated:")
    print("Passage CSV :", passage_csv)
    print("Summary CSV :", summary_csv)
    print("Report      :", markdown_report)

    print("\nSource-level comparison")
    print("-" * 80)

    for config in CONFIGURATIONS:
        config_records = [
            record
            for record in all_records
            if record["configuration"] == config["name"]
        ]

        metrics = summarize_question_records(
            config_records
        )

        print(f"\n{config['name']}")
        print(
            "Source Hit@5:",
            f"{metrics['source_hit_at_5']:.4f}",
        )
        print(
            "Top-1 expected-source rate:",
            f"{metrics['top1_expected_source_rate']:.4f}",
        )
        print(
            "Avg expected-source passages@5:",
            f"{metrics['average_expected_source_count_at_5']:.2f}",
        )
        print(
            "Avg top-1 distance:",
            f"{metrics['average_top1_distance']:.4f}",
        )

    print("\nExisting experiment collections were not modified.")


if __name__ == "__main__":
    main()