
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CANDIDATES_FILE = PROJECT_ROOT / "test" / "passage_candidates.json"
ANNOTATIONS_FILE = PROJECT_ROOT / "test" / "passage_annotations.json"


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    temporary_path = path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    temporary_path.replace(path)


def main():

    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError("Run candidate generation first.")

    candidates = load_json(CANDIDATES_FILE)

    if ANNOTATIONS_FILE.exists():
        annotations = load_json(ANNOTATIONS_FILE)
    else:
        annotations = {}

    total = 0
    completed = 0

    print("=" * 75)
    print("PASSAGE-LEVEL MANUAL ANNOTATION")
    print("=" * 75)

    print("\nRelevance grades:")
    print("0 = Irrelevant")
    print("1 = Partially relevant / supporting")
    print("2 = Directly answers the question")
    print("s = Skip")
    print("q = Save and quit")

    for config_name, questions in candidates["configurations"].items():

        for question_id, question_data in questions.items():

            question = question_data["question"]

            for passage in question_data["passages"]:

                total += 1

                key = (
                    f"{config_name}|{question_id}|"
                    f"{passage['rank']}"
                )

                if key in annotations:
                    completed += 1
                    continue

                print("\n" + "=" * 75)
                print(f"Configuration: {config_name}")
                print(f"Question ID: {question_id}")
                print(f"Question: {question}")
                print("-" * 75)

                print("Rank:", passage["rank"])
                print("Source:", passage["source"])
                print("Page:", passage["page"])
                print("Distance:", passage["distance"])

                print("\nPASSAGE:")
                print(passage["text"])

                while True:

                    choice = input(
                        "\nRelevance grade (0/1/2, s, q): "
                    ).strip().lower()

                    if choice == "q":
                        save_json(ANNOTATIONS_FILE, annotations)
                        print("\nProgress saved. You can resume later.")
                        return

                    if choice == "s":
                        break

                    if choice in {"0", "1", "2"}:

                        annotations[key] = {
                            "configuration": config_name,
                            "question_id": question_id,
                            "rank": passage["rank"],
                            "chunk_id": passage["chunk_id"],
                            "relevance_grade": int(choice)
                        }

                        save_json(ANNOTATIONS_FILE, annotations)

                        completed += 1
                        print("Label saved.")
                        break

                    print("Invalid input. Enter 0, 1, 2, s, or q.")

    print("\n" + "=" * 75)
    print("ANNOTATION SESSION COMPLETED")
    print("=" * 75)

    print("Total passages:", total)
    print("Labeled passages:", completed)
    print("Remaining passages:", total - completed)
    print("Saved to:", ANNOTATIONS_FILE)


if __name__ == "__main__":
    main()

