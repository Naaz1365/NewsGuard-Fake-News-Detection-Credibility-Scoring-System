
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# Create directories
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# LIAR dataset columns
columns = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job_title", "state_info", "party_affiliation",
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context"
]

# Load datasets
print("Loading LIAR datasets...")

train = pd.read_csv("data/raw/train.tsv", sep="\t", names=columns)
valid = pd.read_csv("data/raw/valid.tsv", sep="\t", names=columns)
test = pd.read_csv("data/raw/test.tsv", sep="\t", names=columns)

print("Train:", train.shape)
print("Validation:", valid.shape)
print("Test:", test.shape)

# Combine datasets
df = pd.concat([train, valid, test], ignore_index=True)

print("\nOriginal label distribution:")
print(df["label"].value_counts())

# Convert six labels into two classes
label_mapping = {
    "true": "REAL",
    "mostly-true": "REAL",
    "false": "FAKE",
    "pants-fire": "FAKE"
}

df = df[df["label"].isin(label_mapping)].copy()
df["label"] = df["label"].map(label_mapping)

# Clean missing and duplicate records
df.dropna(subset=["statement", "label"], inplace=True)
df["statement"] = df["statement"].astype(str).str.strip()
df = df[df["statement"] != ""]
df.drop_duplicates(subset=["statement"], inplace=True)
df.reset_index(drop=True, inplace=True)

print("\nCleaned dataset shape:", df.shape)
print("\nBinary label distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df[["statement", "label"]].isnull().sum())

# Stratified 80/10/10 split
train_data, temp_data = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.50,
    random_state=42,
    stratify=temp_data["label"]
)

# Save datasets
train_data.to_csv("data/processed/train.csv", index=False)
val_data.to_csv("data/processed/validation.csv", index=False)
test_data.to_csv("data/processed/test.csv", index=False)

print("\nFinal dataset split:")
print("Training:", len(train_data))
print("Validation:", len(val_data))
print("Testing:", len(test_data))

# Plot binary class distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="label")
plt.title("NewsGuard - Real vs Fake News Distribution")
plt.xlabel("News Category")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig("outputs/plots/class_distribution.png")
plt.close()

# Plot original six labels
original_df = pd.concat([train, valid, test], ignore_index=True)

plt.figure(figsize=(10, 5))
sns.countplot(data=original_df, x="label")
plt.title("Original LIAR Dataset Distribution")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/plots/original_labels.png")
plt.close()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Processed datasets saved.")
print("EDA graphs saved.")
