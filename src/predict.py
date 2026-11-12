
import re
import joblib
import numpy as np

from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec
from scipy.sparse import csr_matrix, hstack

from textstat import flesch_reading_ease
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
FEATURE_DIR = BASE_DIR / "data" / "processed" / "features"

# Load trained model and feature extractors
model = joblib.load(MODEL_PATH)

vectorizer = joblib.load(
    FEATURE_DIR / "tfidf_vectorizer.joblib"
)

word2vec = Word2Vec.load(
    str(FEATURE_DIR / "word2vec.model")
)

# NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
sentiment_analyzer = SentimentIntensityAnalyzer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(tokens)


def get_word2vec_features(cleaned):
    tokens = cleaned.split()

    vectors = [
        word2vec.wv[word]
        for word in tokens
        if word in word2vec.wv
    ]

    if vectors:
        return np.mean(vectors, axis=0).reshape(1, -1)

    return np.zeros((1, word2vec.vector_size), dtype=np.float32)


def get_auxiliary_features(original_text):
    text = original_text.strip()

    if not text:
        return np.zeros((1, 2), dtype=np.float32)

    readability = flesch_reading_ease(text)

    sentiment = sentiment_analyzer.polarity_scores(
        text
    )["compound"]

    return np.array(
        [[readability, sentiment]],
        dtype=np.float32
    )


def predict_news(statement):

    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("Please enter a valid news statement.")

    cleaned = clean_text(statement)

    # 1. TF-IDF features
    tfidf_features = vectorizer.transform([cleaned])

    # 2. Word2Vec features
    word_features = get_word2vec_features(cleaned)

    # 3. Readability + Sentiment features
    auxiliary_features = get_auxiliary_features(statement)

    # Combine features in the same order as training
    combined_features = hstack([
        tfidf_features,
        csr_matrix(word_features),
        csr_matrix(auxiliary_features)
    ]).tocsr()

    expected_features = getattr(model, "n_features_in_", None)

    if (
        expected_features is not None
        and combined_features.shape[1] != expected_features
    ):
        raise ValueError(
            f"Feature mismatch: model expects {expected_features}, "
            f"received {combined_features.shape[1]}"
        )

    # Prediction
    prediction = model.predict(combined_features)[0]
    probabilities = model.predict_proba(combined_features)[0]

    classes = list(model.classes_)

    real_index = classes.index("REAL")
    fake_index = classes.index("FAKE")

    real_probability = float(probabilities[real_index])
    fake_probability = float(probabilities[fake_index])

    credibility_score = round(real_probability * 100, 2)

    return {
        "prediction": prediction,
        "credibility_score": credibility_score,
        "real_probability": round(real_probability * 100, 2),
        "fake_probability": round(fake_probability * 100, 2),
        "readability_score": round(
            float(auxiliary_features[0][0]), 2
        ),
        "sentiment_score": round(
            float(auxiliary_features[0][1]), 4
        ),
        "explanation": (
            "Model predicts this statement as REAL."
            if prediction == "REAL"
            else "Model predicts this statement as FAKE."
        ),
        "disclaimer": (
            "Educational model output only. "
            "The credibility score is model-estimated and "
            "does not verify factual accuracy."
        )
    }


if __name__ == "__main__":

    print("NEWSGUARD - FAKE NEWS DETECTOR")

    statement = input("\nEnter news statement: ")

    try:
        result = predict_news(statement)

        print("\nPrediction:", result["prediction"])
        print(
            "Credibility Score:",
            result["credibility_score"], "/100"
        )
        print("REAL Probability:", result["real_probability"], "%")
        print("FAKE Probability:", result["fake_probability"], "%")
        print("Readability Score:", result["readability_score"])
        print("Sentiment Score:", result["sentiment_score"])
        print("Explanation:", result["explanation"])
        print("Disclaimer:", result["disclaimer"])

    except Exception as error:
        print("\nPrediction Error:", error)
