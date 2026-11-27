
"""
LLM-Assisted Relevance Annotation
=================================

Modes:
    calibrate -> Evaluate LLM against completed human annotations.
    pending   -> Generate provisional labels for Q04 and Q05.

LLM predictions are never written into human ground truth.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd

from genai.llm_service import generate_response


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT / "test" / "reports"
    / "query_strategy_experiment"
    / "query_strategy_review_completed.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "test" / "reports"
    / "query_strategy_experiment"
)

CALIBRATION_CSV = OUTPUT_DIR / "llm_judge_calibration.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "llm_assisted_predictions.csv"


# ============================================================
# QUESTIONS
# ============================================================

QUESTIONS = {
    "Q01": "What are the characteristics of glioma?",
    "Q02": "What are the characteristics of meningioma?",
    "Q03": "What are the characteristics of pituitary tumor?",
    "Q04": "What is a brain MRI used for?",
    "Q05": "What are common symptoms of brain tumors?",
}


# ============================================================
# JUDGE INSTRUCTIONS
# ============================================================

SYSTEM_PROMPT = """
You are a strict, impartial relevance evaluator for a medical
information retrieval benchmark.

Your task is to grade the usefulness of ONE passage for answering
ONE specific question.

GRADING RUBRIC:

GRADE 0 — IRRELEVANT
- Does not meaningfully help answer the question.
- Discusses a different tumor or unrelated medical topic.
- Contains only coincidental keyword overlap.
- Discusses general medical information without meaningful
  relevance to the question.

GRADE 1 — RELATED BUT INSUFFICIENT
- Discusses the same topic but provides only peripheral,
  indirect, or incomplete information.
- Mentions risk factors, demographics, prognosis, or treatment
  without providing the characteristics being asked about.
- Mentions a related concept but does not explain it sufficiently.
- Provides background context rather than a substantive answer.

GRADE 2 — DIRECTLY USEFUL
- Contains concrete factual information that directly answers
  at least one substantive part of the question.
- Provides actual characteristics, explanations, symptoms,
  diagnostic uses, or other information specifically requested.
- The reader could use the passage as evidence when answering
  the question without inventing missing information.

CRITICAL DISTINCTIONS:

1. Grade 2 is NOT awarded merely because a passage discusses
   the same disease or contains matching keywords.

2. For questions asking about tumor characteristics, information
   about origin, behavior, location, grade, or defining features
   can support Grade 2. Risk factors or demographics alone
   generally support Grade 1 at most.

3. For questions asking about symptoms, actual symptoms or
   their effects support Grade 2. Merely discussing tumor types
   or diagnosis does not.

4. For questions asking what MRI is used for, actual diagnostic
   applications support Grade 2. Merely explaining MRI technology
   does not automatically qualify.

5. A passage may receive Grade 2 even if it does not answer
   the entire question, provided it directly answers a substantive
   part of it.

6. Do not reward retrieval rank, source filename, or keyword
   frequency.

7. Be conservative when evidence is ambiguous. Prefer Grade 1
   over Grade 2 when direct usefulness is not established.

8. Confidence describes certainty in the assigned grade,
   not the quality of the passage.

Return ONLY one valid JSON object.

Required schema:
{
    "grade": 0,
    "confidence": "high",
    "reason": "Brief explanation citing the actual passage content."
}

Grade must be an integer: 0, 1, or 2.
Confidence must be exactly: high, medium, or low.
"""


RETRY_SYSTEM_PROMPT = """
Return exactly one valid JSON object and nothing else.

Schema:
{"grade": 0, "confidence": "high", "reason": "Brief explanation"}

grade must be integer 0, 1, or 2.
confidence must be high, medium, or low.
reason must be a non-empty string.

Do not use Markdown fences or additional commentary.
"""


# ============================================================
# HELPERS
# ============================================================

def passage_identity(row):
    """Create an identity that protects against passage ID collisions."""

    text = str(row.get("passage_text", "") or "")

    text_hash = hashlib.sha256(
        text.strip().encode("utf-8")
    ).hexdigest()

    return (
        str(row["question_id"]),
        str(row["passage_id"]),
        str(row["source"]),
        str(row["page"]),
        text_hash,
    )


def extract_json(response):
    """Extract the first valid JSON object from an LLM response."""

    if not isinstance(response, str) or not response.strip():
        raise ValueError("Empty LLM response.")

    decoder = json.JSONDecoder()

    for index, character in enumerate(response):

        if character != "{":
            continue

        try:
            data, _ = decoder.raw_decode(response[index:])

            if isinstance(data, dict):
                return data

        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON object found in response.")


def validate_prediction(response):

    try:
        data = extract_json(response)

        grade = data.get("grade")
        confidence = str(data.get("confidence", "")).lower()
        reason = data.get("reason")

        if isinstance(grade, bool) or not isinstance(grade, int):
            raise ValueError("Grade must be an integer.")

        if grade not in (0, 1, 2):
            raise ValueError("Grade must be 0, 1, or 2.")

        if confidence not in ("high", "medium", "low"):
            raise ValueError("Invalid confidence value.")

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Missing explanation.")

        return {
            "predicted_grade": grade,
            "llm_confidence": confidence,
            "llm_reason": reason.strip(),
            "prediction_status": "SUCCESS",
        }

    except Exception as error:

        return {
            "predicted_grade": None,
            "llm_confidence": None,
            "llm_reason": str(error),
            "prediction_status": "PARSE_ERROR",
        }


# ============================================================
# LLM JUDGMENT
# ============================================================

def judge_passage(question, passage):

    if not isinstance(passage, str) or not passage.strip():
        raise ValueError("Passage text is empty.")

    prompt = f"""
QUESTION:
{question}

PASSAGE:
{passage}

Apply the grading rubric strictly.

Before assigning Grade 2, verify that the passage contains
concrete information directly answering a substantive part
of the question.

Return only the required JSON object.
"""

    response = generate_response(
        user_prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0,
        max_tokens=300,
    )

    prediction = validate_prediction(response)

    # Retry once if the response is not valid JSON.
    if prediction["prediction_status"] != "SUCCESS":

        retry_prompt = f"""
The previous response was not valid JSON.

Evaluate the following passage again.

QUESTION:
{question}

PASSAGE:
{passage}

Return exactly one valid JSON object.
"""

        response = generate_response(
            user_prompt=retry_prompt,
            system_prompt=RETRY_SYSTEM_PROMPT,
            temperature=0,
            max_tokens=300,
        )

        prediction = validate_prediction(response)

    return prediction


# ============================================================
# CALIBRATION MODE
# ============================================================

def run_calibration(df):

    calibration = df[
        df["question_id"].isin(["Q01", "Q02", "Q03"])
        & df["verified_relevance_grade"].notna()
    ].copy()

    if calibration.empty:
        raise ValueError("No completed human annotations found.")

    results = []

    print("=" * 75)
    print("LLM JUDGE CALIBRATION — STRICT RUBRIC")
    print("=" * 75)

    print(f"Human-labeled passages: {len(calibration)}")

    for index, (_, row) in enumerate(
        calibration.iterrows(), start=1
    ):

        question_id = row["question_id"]

        print(
            f"\n[{index}/{len(calibration)}] "
            f"{question_id} | {row['passage_id']}"
        )

        try:
            prediction = judge_passage(
                QUESTIONS[question_id],
                row["passage_text"],
            )

        except Exception as error:
            prediction = {
                "predicted_grade": None,
                "llm_confidence": None,
                "llm_reason": str(error),
                "prediction_status": "API_ERROR",
            }

        result = {
            "question_id": question_id,
            "passage_id": row["passage_id"],
            "source": row["source"],
            "page": row["page"],
            "human_grade": int(row["verified_relevance_grade"]),
            **prediction,
        }

        if prediction["prediction_status"] == "SUCCESS":

            result["exact_match"] = (
                result["human_grade"]
                == result["predicted_grade"]
            )

            result["absolute_grade_difference"] = abs(
                result["human_grade"]
                - result["predicted_grade"]
            )

            print(
                f"Human: {result['human_grade']} | "
                f"LLM: {result['predicted_grade']} | "
                f"Confidence: {result['llm_confidence']}"
            )

        else:
            result["exact_match"] = None
            result["absolute_grade_difference"] = None

            print("Prediction failed:", prediction["llm_reason"])

        results.append(result)

        pd.DataFrame(results).to_csv(
            CALIBRATION_CSV,
            index=False,
            encoding="utf-8",
        )

        time.sleep(1)

    output = pd.DataFrame(results)

    successful = output[
        output["prediction_status"].eq("SUCCESS")
    ]

    print("\n" + "=" * 75)
    print("CALIBRATION RESULTS")
    print("=" * 75)

    print(f"Total passages: {len(output)}")
    print(f"Successful predictions: {len(successful)}")
    print(f"Failed predictions: {len(output) - len(successful)}")

    if not successful.empty:

        print(
            "Exact agreement:",
            f"{successful['exact_match'].mean():.2%}",
        )

        print(
            "Within-one-grade agreement:",
            f"{(successful['absolute_grade_difference'] <= 1).mean():.2%}",
        )

        print("\nConfusion matrix:")

        print(
            pd.crosstab(
                successful["human_grade"],
                successful["predicted_grade"],
                rownames=["Human"],
                colnames=["LLM"],
            ).reindex(
                index=[0, 1, 2],
                columns=[0, 1, 2],
                fill_value=0,
            )
        )

        print("\nAgreement by human grade:")

        for grade in [0, 1, 2]:

            subset = successful[
                successful["human_grade"] == grade
            ]

            if not subset.empty:

                agreement = (
                    subset["human_grade"]
                    == subset["predicted_grade"]
                ).mean()

                print(
                    f"Grade {grade}: "
                    f"{agreement:.2%} "
                    f"({len(subset)} passages)"
                )

    print(f"\nSaved to:\n{CALIBRATION_CSV}")


# ============================================================
# PENDING ANNOTATION MODE
# ============================================================

def run_pending(df):

    pending = df[
        df["question_id"].isin(["Q04", "Q05"])
        & df["verified_relevance_grade"].isna()
    ].copy()

    if pending.empty:
        print("No pending Q04/Q05 passages found.")
        return

    # Load existing predictions for safe resume.
    if PREDICTIONS_CSV.exists():

        existing = pd.read_csv(PREDICTIONS_CSV)

        completed = set()

        for _, row in existing.iterrows():

            if row.get("prediction_status") == "SUCCESS":
                completed.add(passage_identity(row))

        pending = pending[
            ~pending.apply(
                lambda row: passage_identity(row) in completed,
                axis=1,
            )
        ]

        results = existing.to_dict("records")

    else:
        results = []

    if pending.empty:
        print("All pending passages already have predictions.")
        return

    print("=" * 75)
    print("LLM-ASSISTED PENDING ANNOTATION")
    print("=" * 75)

    print(f"New passages to classify: {len(pending)}")

    for index, (_, row) in enumerate(
        pending.iterrows(), start=1
    ):

        question_id = row["question_id"]

        print(
            f"\n[{index}/{len(pending)}] "
            f"{question_id} | {row['passage_id']}"
        )

        try:
            prediction = judge_passage(
                QUESTIONS[question_id],
                row["passage_text"],
            )

        except Exception as error:
            prediction = {
                "predicted_grade": None,
                "llm_confidence": None,
                "llm_reason": str(error),
                "prediction_status": "API_ERROR",
            }

        result = {
            "question_id": question_id,
            "passage_id": row["passage_id"],
            "source": row["source"],
            "page": row["page"],
            "best_rank": row["best_rank"],
            "passage_text": row["passage_text"],
            "passage_text_hash": passage_identity(row)[-1],
            **prediction,
            "annotation_source": "LLM_PROVISIONAL",
            "human_review_status": "NOT_REVIEWED",
        }

        results.append(result)

        if prediction["prediction_status"] == "SUCCESS":

            print(
                f"Predicted grade: {prediction['predicted_grade']} | "
                f"Confidence: {prediction['llm_confidence']}"
            )

        else:
            print("Prediction failed:", prediction["llm_reason"])

        pd.DataFrame(results).to_csv(
            PREDICTIONS_CSV,
            index=False,
            encoding="utf-8",
        )

        time.sleep(1)

    output = pd.DataFrame(results)

    print("\n" + "=" * 75)
    print("PROVISIONAL ANNOTATION COMPLETED")
    print("=" * 75)

    print(f"Total predictions saved: {len(output)}")

    print(
        "Successful:",
        output["prediction_status"].eq("SUCCESS").sum(),
    )

    print(
        "Failed:",
        output["prediction_status"].ne("SUCCESS").sum(),
    )

    print(f"\nSaved to:\n{PREDICTIONS_CSV}")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["calibrate", "pending"],
        required=True,
    )

    args = parser.parse_args()

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "question_id",
        "passage_id",
        "passage_text",
        "verified_relevance_grade",
        "source",
        "page",
        "best_rank",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "calibrate":
        run_calibration(df)

    elif args.mode == "pending":
        run_pending(df)


if __name__ == "__main__":
    main()
