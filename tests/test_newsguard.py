
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_model_file_exists():
    assert (ROOT / "models" / "best_model.joblib").exists()


def test_tfidf_vectorizer_exists():
    assert (
        ROOT / "data" / "processed" / "features" / "tfidf_vectorizer.joblib"
    ).exists()


def test_word2vec_model_exists():
    assert (
        ROOT / "data" / "processed" / "features" / "word2vec.model"
    ).exists()


def test_processed_datasets_exist():
    for name in ["train_cleaned.csv", "validation_cleaned.csv", "test_cleaned.csv"]:
        assert (ROOT / "data" / "processed" / name).exists()


def test_model_prediction():
    import joblib

    model = joblib.load(ROOT / "models" / "best_model.joblib")

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_model_probability_output():
    import joblib
    from scipy.sparse import csr_matrix

    model = joblib.load(ROOT / "models" / "best_model.joblib")

    sample = csr_matrix((1, model.n_features_in_))
    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (1, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
