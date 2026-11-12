
import os
import joblib
import numpy as np

from scipy.sparse import load_npz
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = os.path.join(BASE, "data", "processed", "features")
MODELS = os.path.join(BASE, "models")

X_train = load_npz(os.path.join(FEATURES, "X_train.npz"))
X_val = load_npz(os.path.join(FEATURES, "X_validation.npz"))

y_train = np.load(os.path.join(FEATURES, "y_train.npy"))
y_val = np.load(os.path.join(FEATURES, "y_validation.npy"))

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)

model = ExtraTreesClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [20, None],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=3,
    n_jobs=1,
    verbose=1
)

print("\nStarting GridSearchCV...")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
predictions = best_model.predict(X_val)

fake_label = "FAKE" if "FAKE" in best_model.classes_ else best_model.classes_[0]

print("\nBest Parameters:", grid.best_params_)
print("Best CV F1 Macro:", grid.best_score_)

print("\nValidation Results:")
print("Accuracy:", accuracy_score(y_val, predictions))
print("Precision FAKE:", precision_score(
    y_val, predictions, pos_label=fake_label, zero_division=0
))
print("Recall FAKE:", recall_score(
    y_val, predictions, pos_label=fake_label, zero_division=0
))
print("F1 FAKE:", f1_score(
    y_val, predictions, pos_label=fake_label, zero_division=0
))

print("\nClassification Report:")
print(classification_report(y_val, predictions))

joblib.dump(
    best_model,
    os.path.join(MODELS, "tuned_extra_trees.joblib")
)

print("\nTuned model saved successfully!")
