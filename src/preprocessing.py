
import csv
import re
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = word_tokenize(text)

    cleaned_tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in STOP_WORDS
    ]

    return " ".join(cleaned_tokens)


def process_file(filename):
    input_path = DATA_DIR / filename
    output_path = DATA_DIR / filename.replace(".csv", "_cleaned.csv")

    with open(input_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames or "statement" not in fieldnames:
        raise ValueError(f"'statement' column missing in {filename}")

    for row in rows:
        row["cleaned_statement"] = clean_text(row["statement"])

    output_fields = list(fieldnames)
    output_fields.append("cleaned_statement")

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed: {filename}")
    print(f"Saved: {output_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    for filename in ["train.csv", "validation.csv", "test.csv"]:
        process_file(filename)

    print("\nPreprocessing completed successfully!")
