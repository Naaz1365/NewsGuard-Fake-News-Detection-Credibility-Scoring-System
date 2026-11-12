
import csv
import json
import numpy as np
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack, save_npz
from gensim.models import Word2Vec
from textstat import flesch_reading_ease
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
FEATURE_DIR = DATA_DIR / "features"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

sentiment_analyzer = SentimentIntensityAnalyzer()


def load_data(filename):
    path = DATA_DIR / filename

    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    cleaned_texts = [row["cleaned_statement"] or "" for row in rows]
    original_texts = [row["statement"] or "" for row in rows]
    labels = [row["label"] for row in rows]

    return cleaned_texts, original_texts, labels


def train_word2vec(train_texts):
    sentences = [text.split() for text in train_texts]

    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=1,
        sg=1,
        epochs=10,
        seed=42
    )

    return model


def get_word2vec_features(texts, model):
    features = []

    for text in texts:
        words = text.split()

        vectors = [
            model.wv[word]
            for word in words
            if word in model.wv
        ]

        if vectors:
            features.append(np.mean(vectors, axis=0))
        else:
            features.append(np.zeros(model.vector_size))

    return np.array(features, dtype=np.float32)


def get_auxiliary_features(texts):
    features = []

    for text in texts:
        text = text.strip()

        if text:
            readability = flesch_reading_ease(text)
            sentiment = sentiment_analyzer.polarity_scores(text)["compound"]
        else:
            readability = 0.0
            sentiment = 0.0

        features.append([readability, sentiment])

    return np.array(features, dtype=np.float32)


def main():
    print("Loading datasets...")

    X_train, raw_train, y_train = load_data("train_cleaned.csv")
    X_val, raw_val, y_val = load_data("validation_cleaned.csv")
    X_test, raw_test, y_test = load_data("test_cleaned.csv")

    print("Training TF-IDF...")

    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    print("Training Word2Vec...")

    word2vec = train_word2vec(X_train)

    X_train_w2v = get_word2vec_features(X_train, word2vec)
    X_val_w2v = get_word2vec_features(X_val, word2vec)
    X_test_w2v = get_word2vec_features(X_test, word2vec)

    print("Calculating readability and sentiment...")

    X_train_aux = get_auxiliary_features(raw_train)
    X_val_aux = get_auxiliary_features(raw_val)
    X_test_aux = get_auxiliary_features(raw_test)

    print("Combining all features...")

    X_train_final = hstack([
        X_train_tfidf,
        csr_matrix(X_train_w2v),
        csr_matrix(X_train_aux)
    ]).tocsr()

    X_val_final = hstack([
        X_val_tfidf,
        csr_matrix(X_val_w2v),
        csr_matrix(X_val_aux)
    ]).tocsr()

    X_test_final = hstack([
        X_test_tfidf,
        csr_matrix(X_test_w2v),
        csr_matrix(X_test_aux)
    ]).tocsr()

    print("Saving feature matrices...")

    save_npz(FEATURE_DIR / "X_train.npz", X_train_final)
    save_npz(FEATURE_DIR / "X_validation.npz", X_val_final)
    save_npz(FEATURE_DIR / "X_test.npz", X_test_final)

    np.save(FEATURE_DIR / "y_train.npy", np.array(y_train))
    np.save(FEATURE_DIR / "y_validation.npy", np.array(y_val))
    np.save(FEATURE_DIR / "y_test.npy", np.array(y_test))

    joblib.dump(tfidf, FEATURE_DIR / "tfidf_vectorizer.joblib")
    word2vec.save(str(FEATURE_DIR / "word2vec.model"))

    with open(FEATURE_DIR / "auxiliary_features.json", "w") as file:
        json.dump(
            ["readability_flesch_score", "sentiment_compound"],
            file,
            indent=4
        )

    print("\nFEATURE EXTRACTION COMPLETED!")
    print("Train shape:", X_train_final.shape)
    print("Validation shape:", X_val_final.shape)
    print("Test shape:", X_test_final.shape)
    print("Additional features: Readability + Sentiment")
    print("Feature files saved in:", FEATURE_DIR)


if __name__ == "__main__":
    main()
