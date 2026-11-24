# test/annotate_failure_causes.py

import csv
from pathlib import Path

CSV_PATH = Path("test/reports/failure_cause_annotations.csv")

# Key format: configuration | question_id | rank | chunk_id
ANNOTATIONS = {
    # ---------------- Batch 1: Records 1-10 ----------------
    ("A", "glioma", 1, "chunk_32"): (
        "related_not_answer",
        "Describes astrocytoma subtypes but does not directly define glioma.",
        "medium",
    ),
    ("A", "glioma", 2, "chunk_31"): (
        "related_not_answer",
        "Mentions gliomas in relation to glial cells but provides limited definition context.",
        "medium",
    ),
    ("A", "glioma", 3, "chunk_74"): (
        "wrong_section",
        "Discusses brain stem glioma treatment and PET imaging rather than defining glioma.",
        "high",
    ),
    ("A", "glioma", 4, "chunk_33"): (
        "related_not_answer",
        "Describes oligodendroglial and mixed glioma subtypes rather than a general definition.",
        "high",
    ),
    ("A", "glioma", 5, "chunk_22"): (
        "related_not_answer",
        "General CNS tumor overview and classification without a direct glioma definition.",
        "medium",
    ),
    ("A", "meningioma", 1, "chunk_100"): (
        "wrong_section",
        "Focuses on symptoms, demographics and prevalence rather than defining meningioma.",
        "high",
    ),
    ("A", "meningioma", 4, "chunk_97"): (
        "related_not_answer",
        "Discusses meningioma grading and imaging rather than its general definition.",
        "medium",
    ),
    ("A", "pituitary", 2, "chunk_114"): (
        "related_not_answer",
        "Discusses pituitary tumor spread, imaging, size and grading rather than defining it.",
        "medium",
    ),
    ("A", "pituitary", 3, "chunk_115"): (
        "related_not_answer",
        "Describes tumor size categories and grading, not a direct definition.",
        "high",
    ),
    ("A", "pituitary", 4, "chunk_122"): (
        "wrong_section",
        "Focuses on hormone-related symptoms instead of defining pituitary tumors.",
        "high",
    ),

    # Record 11 intentionally omitted.
    # Actual passage text for this record was not supplied.

    # ---------------- Batch 2: Records 12-20 ----------------
    ("A", "mri", 2, "chunk_4"): (
        "related_not_answer",
        "Explains MRI benefits and clinical uses but does not provide a concise general definition.",
        "high",
    ),
    ("A", "mri", 4, "chunk_107"): (
        "related_not_answer",
        "Defines MRI with gadolinium in a pituitary diagnostic context.",
        "high",
    ),
    ("A", "mri", 5, "chunk_10"): (
        "boilerplate_noise",
        "Contains website disclaimers, copyright and publication metadata.",
        "high",
    ),
    ("A", "symptoms", 2, "chunk_42"): (
        "related_not_answer",
        "Lists spinal cord tumor symptoms rather than general brain tumor symptoms.",
        "high",
    ),
    ("A", "symptoms", 3, "chunk_23"): (
        "related_not_answer",
        "Contains general CNS tumor information and risk context but no symptom list.",
        "high",
    ),
    ("A", "symptoms", 4, "chunk_25"): (
        "related_not_answer",
        "Explains tumor behavior and symptom effects but does not list common symptoms.",
        "high",
    ),
    ("A", "symptoms", 5, "chunk_26"): (
        "related_not_answer",
        "Focuses on metastatic tumor information rather than symptom descriptions.",
        "high",
    ),
    ("B", "glioma", 1, "chunk_41"): (
        "related_not_answer",
        "Describes glioma-related tumor subtypes without providing a general glioma definition.",
        "high",
    ),
    ("B", "glioma", 3, "chunk_42"): (
        "related_not_answer",
        "Describes oligodendroglioma and mixed glioma subtypes rather than defining glioma generally.",
        "high",
    ),

    # ---------------- Batch 3: Records 21-30 ----------------
    ("B", "glioma", 4, "chunk_26"): (
        "related_not_answer",
        "Provides general CNS tumor definitions and tumor type headings, but no direct glioma definition.",
        "high",
    ),
    ("B", "glioma", 5, "chunk_104"): (
        "wrong_section",
        "Discusses glioblastoma treatment options rather than defining glioma.",
        "high",
    ),
    ("B", "meningioma", 1, "chunk_131"): (
        "related_not_answer",
        "Contains survivor-story material, imaging and grade information, not a direct definition.",
        "high",
    ),
    ("B", "meningioma", 4, "chunk_136"): (
        "incomplete_context",
        "Fragment contains a survival-rate caption and page metadata without useful definition context.",
        "high",
    ),
    ("B", "meningioma", 5, "chunk_138"): (
        "wrong_section",
        "Describes treatment of atypical and anaplastic meningiomas.",
        "high",
    ),
    ("B", "pituitary", 1, "chunk_153"): (
        "wrong_section",
        "Focuses on pituitary tumor size and grade rather than its definition.",
        "high",
    ),
    ("B", "pituitary", 2, "chunk_157"): (
        "related_not_answer",
        "Explains pituitary tumor categories and characteristics but lacks a direct general definition.",
        "high",
    ),
    ("B", "pituitary", 4, "chunk_152"): (
        "related_not_answer",
        "Includes MRI information and tumor size and grade classification, only partially answering the question.",
        "high",
    ),
    ("B", "pituitary", 5, "chunk_164"): (
        "related_not_answer",
        "Focuses on symptoms caused by functioning pituitary tumors rather than defining them.",
        "high",
    ),
    ("B", "mri", 4, "chunk_143"): (
        "related_not_answer",
        "Contains neurological examination information and an MRI description in a diagnostic context.",
        "high",
    ),

    # ---------------- Batch 4: Records 31-40 ----------------
    ("B", "mri", 5, "chunk_6"): (
        "related_not_answer",
        "Describes MRI benefits and diagnostic uses, but provides only partial coverage of the query.",
        "high",
    ),
    ("B", "symptoms", 3, "chunk_27"): (
        "related_not_answer",
        "Lists tumor types and general information but does not provide symptom details.",
        "high",
    ),
    ("B", "symptoms", 4, "chunk_95"): (
        "wrong_section",
        "Focuses on follow-up testing and monitoring treatment response.",
        "high",
    ),
    ("B", "symptoms", 5, "chunk_29"): (
        "related_not_answer",
        "Explains tumor behavior and that tumors cause symptoms, but does not list common symptoms.",
        "high",
    ),
    ("C", "glioma", 1, "chunk_49"): (
        "related_not_answer",
        "Describes brain stem glioma and pineal astrocytic tumors rather than defining glioma generally.",
        "high",
    ),
    ("C", "glioma", 2, "chunk_54"): (
        "related_not_answer",
        "Describes oligodendroglial and mixed glioma subtypes rather than a general definition.",
        "high",
    ),
    ("C", "glioma", 3, "chunk_52"): (
        "related_not_answer",
        "Describes astrocytoma and glioblastoma subtypes, not a general glioma definition.",
        "high",
    ),
    ("C", "glioma", 5, "chunk_99"): (
        "wrong_section",
        "Discusses treatment considerations for tumors metastatic to the brain.",
        "high",
    ),
    ("C", "meningioma", 1, "chunk_191"): (
        "wrong_section",
        "Focuses on treatment of malignant meningiomas rather than defining the tumor.",
        "high",
    ),
    ("C", "meningioma", 2, "chunk_181"): (
        "related_not_answer",
        "Contains survivor-story material and meningioma grade and imaging details rather than a direct definition.",
        "high",
    ),

    # ---------------- Batch 5: Records 41-50 ----------------
    ("C", "meningioma", 4, "chunk_182"): (
        "related_not_answer",
        "Focuses on meningioma causes and risk factors rather than defining the tumor.",
        "high",
    ),
    ("C", "pituitary", 1, "chunk_215"): (
        "related_not_answer",
        "Explains pituitary tumor categories and characteristics, providing partial definition context.",
        "high",
    ),
    ("C", "pituitary", 2, "chunk_209"): (
        "wrong_section",
        "Focuses on tumor spread and staging rather than defining pituitary tumors.",
        "high",
    ),
    ("C", "pituitary", 3, "chunk_210"): (
        "wrong_section",
        "Describes MRI, tumor size categories and grading rather than a direct definition.",
        "high",
    ),
    ("C", "pituitary", 4, "chunk_216"): (
        "related_not_answer",
        "Describes pituitary tumor types and functionality, providing related classification information.",
        "high",
    ),
    ("C", "mri", 1, "chunk_3"): (
        "related_not_answer",
        "Describes MRI diagnostic uses for brain conditions but only partially answers the general MRI question.",
        "high",
    ),
    ("C", "symptoms", 2, "chunk_35"): (
        "related_not_answer",
        "Defines general brain and spinal cord tumors and their behavior but does not list symptoms.",
        "high",
    ),
    ("C", "symptoms", 3, "chunk_69"): (
        "related_not_answer",
        "Explains factors affecting tumor symptoms but does not provide the actual symptom list.",
        "high",
    ),
    ("C", "symptoms", 4, "chunk_36"): (
        "related_not_answer",
        "Describes tumor behavior and effects but does not identify common symptoms.",
        "high",
    ),
    ("C", "symptoms", 5, "chunk_43"): (
        "incomplete_context",
        "Fragment of brain anatomy with extraction metadata and no useful symptom information.",
        "high",
    ),
}


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {CSV_PATH}")

    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    required_fields = {
        "configuration",
        "question_id",
        "rank",
        "chunk_id",
        "failure_cause",
        "review_note",
        "review_confidence",
    }

    missing_fields = required_fields - set(fieldnames or [])
    if missing_fields:
        raise ValueError(f"CSV missing required columns: {sorted(missing_fields)}")

    seen = set()
    updated = 0

    for row in rows:
        key = (
            row["configuration"].strip(),
            row["question_id"].strip(),
            int(row["rank"]),
            row["chunk_id"].strip(),
        )

        if key in seen:
            raise ValueError(f"Duplicate row key found: {key}")
        seen.add(key)

        if key in ANNOTATIONS:
            cause, note, confidence = ANNOTATIONS[key]

            row["failure_cause"] = cause
            row["review_note"] = note
            row["review_confidence"] = confidence
            updated += 1

    missing_keys = set(ANNOTATIONS) - seen
    if missing_keys:
        raise ValueError(
            "Annotation keys not found in CSV:\n"
            + "\n".join(str(key) for key in sorted(missing_keys))
        )

    # Write back the same CSV, preserving all rows and original grades.
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    reviewed = sum(
        1 for row in rows
        if row["failure_cause"].strip()
    )
    unreviewed = len(rows) - reviewed

    print("=" * 60)
    print("FAILURE CAUSE ANNOTATIONS UPDATED")
    print("=" * 60)
    print(f"CSV records          : {len(rows)}")
    print(f"New annotations      : {updated}")
    print(f"Reviewed records     : {reviewed}")
    print(f"Unreviewed records   : {unreviewed}")
    print(f"Output               : {CSV_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()