
import pandas as pd
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent

csv_path = (
    BASE_DIR / "reports" / "hybrid_retrieval"
    / "hybrid_relevance_annotations.csv"
)

backup_path = csv_path.with_name(
    "hybrid_relevance_annotations.before_final_fix.csv"
)

df = pd.read_csv(csv_path)

mask = (
    (df["question_id"].astype(str) == "Q04")
    & (df["passage_id"].astype(str) == "chunk_107")
)

if int(mask.sum()) != 1:
    raise ValueError(
        f"Expected exactly one Q04/chunk_107 row, found {int(mask.sum())}. "
        "No changes made."
    )

if backup_path.exists():
    raise FileExistsError(f"Backup already exists: {backup_path}")

shutil.copy2(csv_path, backup_path)

df.loc[mask, "relevance_grade"] = 1
df.loc[mask, "review_note"] = (
    "Describes MRI imaging with gadolinium and visualization of "
    "brain and spinal cord structures, but provides limited "
    "information about broader brain MRI diagnostic applications."
)
df.loc[mask, "review_confidence"] = "high"

df.to_csv(csv_path, index=False)

print("Remaining annotation corrected.")
print(f"Backup: {backup_path}")
