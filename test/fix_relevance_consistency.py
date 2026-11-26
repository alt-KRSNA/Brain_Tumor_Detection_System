
import pandas as pd
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent

csv_path = (
    BASE_DIR / "reports" / "retrieval_baseline"
    / "baseline_relevance_annotations.csv"
)

backup_path = csv_path.with_name(
    "baseline_relevance_annotations.before_consistency_fix.csv"
)

corrections = {
    ("Q01", "chunk_31"): (
        1,
        "Describes astrocytic tumors and examples of glioma subtypes, "
        "but provides only partial information about overall glioma characteristics."
    ),
    ("Q01", "chunk_32"): (
        2,
        "Describes astrocytoma subtypes and glioblastoma, including "
        "grades, growth rates, spread patterns, and tumor-cell appearance."
    ),
    ("Q02", "chunk_97"): (
        2,
        "Describes meningioma grades, growth rates, recurrence risk, "
        "malignant characteristics, and associated subtypes."
    ),
    ("Q03", "chunk_114"): (
        2,
        "Describes pituitary tumor classification by size, grade, "
        "hormone production, and spread."
    ),
    ("Q03", "chunk_115"): (
        2,
        "Defines pituitary tumor size categories and explains grading "
        "based on invasion into surrounding structures."
    ),
    ("Q04", "chunk_107"): (
        2,
        "Explains MRI imaging of the brain and spinal cord and how "
        "gadolinium contrast helps highlight cancer cells."
    ),
    ("Q04", "chunk_6"): (
        0,
        "Focuses on MRI limitations and contrast-related considerations "
        "without describing its diagnostic uses."
    ),
}

df = pd.read_csv(csv_path)

required = {
    "configuration", "question_id", "chunk_id",
    "relevance_grade", "review_note", "review_confidence"
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Select only Dense configuration A
config = df["configuration"].astype(str).str.upper()
mask_a = config.isin(["A", "A - BASELINE", "BASELINE"])

if not mask_a.any():
    raise ValueError("Could not find Dense configuration A.")

# Validate all seven target rows before changing anything
for (qid, chunk_id), (grade, note) in corrections.items():
    mask = (
        mask_a
        & (df["question_id"].astype(str) == qid)
        & (df["chunk_id"].astype(str) == chunk_id)
    )

    count = int(mask.sum())

    if count != 1:
        raise ValueError(
            f"Expected exactly one row for {qid}/{chunk_id}, found {count}. "
            "No changes have been made."
        )

# Create backup before editing
if backup_path.exists():
    raise FileExistsError(
        f"Backup already exists: {backup_path}. "
        "Rename it before running this script again."
    )

shutil.copy2(csv_path, backup_path)

# Apply corrections
for (qid, chunk_id), (grade, note) in corrections.items():
    mask = (
        mask_a
        & (df["question_id"].astype(str) == qid)
        & (df["chunk_id"].astype(str) == chunk_id)
    )

    df.loc[mask, "relevance_grade"] = grade
    df.loc[mask, "review_note"] = note
    df.loc[mask, "review_confidence"] = "high"

df.to_csv(csv_path, index=False)

print("Relevance consistency corrections completed.")
print(f"Corrected rows: {len(corrections)}")
print(f"Backup: {backup_path}")
print(f"Updated CSV: {csv_path}")
