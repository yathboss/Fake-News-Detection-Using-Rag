import os
import pandas as pd

# =========================
# 1) Paths
# =========================
base_folder = r"c:\Users\Yatharth Singh\Downloads\archive (1)"
output_folder = os.path.join(base_folder, "processed_liar")

os.makedirs(output_folder, exist_ok=True)

# =========================
# 2) Column names for LIAR
# =========================
columns = [
    "id",
    "label",
    "statement",
    "subjects",
    "speaker",
    "speaker_job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context"
]

# =========================
# 3) Function to load + clean
# =========================
def load_and_clean_tsv(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None, names=columns)

    # Keep only important columns
    df = df[["label", "statement"]].copy()

    # Remove nulls
    df.dropna(subset=["label", "statement"], inplace=True)

    # Convert to string and clean spaces
    df["label"] = df["label"].astype(str).str.strip()
    df["statement"] = df["statement"].astype(str).str.strip()

    # Remove empty statements
    df = df[df["statement"] != ""]

    return df

# =========================
# 4) File paths
# =========================
train_path = os.path.join(base_folder, "train.tsv")
valid_path = os.path.join(base_folder, "valid.tsv")
test_path = os.path.join(base_folder, "test.tsv")

# =========================
# 5) Load files
# =========================
train_df = load_and_clean_tsv(train_path)
valid_df = load_and_clean_tsv(valid_path)
test_df = load_and_clean_tsv(test_path)

# =========================
# 6) Save cleaned CSVs
# =========================
train_df.to_csv(os.path.join(output_folder, "train_clean.csv"), index=False)
valid_df.to_csv(os.path.join(output_folder, "valid_clean.csv"), index=False)
test_df.to_csv(os.path.join(output_folder, "test_clean.csv"), index=False)

print("Cleaned CSV files saved successfully.")

# =========================
# 7) Show label distribution
# =========================
print("\nTrain label distribution:")
print(train_df["label"].value_counts())

print("\nValid label distribution:")
print(valid_df["label"].value_counts())

print("\nTest label distribution:")
print(test_df["label"].value_counts())

# =========================
# 8) Balanced Phase-1 sample
# =========================
sample_per_class = 100

balanced_sample = (
    train_df.groupby("label", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), sample_per_class), random_state=42))
    .reset_index(drop=True)
)

balanced_sample.to_csv(os.path.join(output_folder, "phase1_balanced_sample.csv"), index=False)

print("\nBalanced Phase-1 sample saved.")
print("Sample size:", len(balanced_sample))
print("\nBalanced sample label distribution:")
print(balanced_sample["label"].value_counts())

print("\nPreview:")
print(balanced_sample.head())