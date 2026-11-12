
import os
import joblib
import numpy as np

from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = os.path.join(BASE, "data", "processed", "features")
MODELS = os.path.join(BASE, "models")

X_val = load_npz(os.path.join(FEATURES, "X_validation.npz"))
y_val = np.load(os.path.join(FEATURES, "y_validation.npy"))

models = {
    "Original Extra Trees": joblib.load(
        os.path.join(MODELS, "best_model.joblib")
    ),
    "Tuned Extra Trees": joblib.load(
        os.path.join(MODELS, "tuned_extra_trees.joblib")
    )
}

for name, model in models.items():
    pred = model.predict(X_val)
    fake_label = "FAKE"

    print(f"\n{name}")
    print("-" * 35)
    print("Accuracy:", round(accuracy_score(y_val, pred), 4))
    print("FAKE Precision:", round(
        precision_score(y_val, pred, pos_label=fake_label, zero_division=0), 4
    ))
    print("FAKE Recall:", round(
        recall_score(y_val, pred, pos_label=fake_label, zero_division=0), 4
    ))
    print("FAKE F1:", round(
        f1_score(y_val, pred, pos_label=fake_label, zero_division=0), 4
    ))
    print("Macro F1:", round(f1_score(y_val, pred, average="macro"), 4))
