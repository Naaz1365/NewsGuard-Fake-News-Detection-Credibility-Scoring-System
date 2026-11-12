
import json
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.sparse import load_npz


BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_DIR = BASE_DIR / "data" / "processed" / "features"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def load_features():
    X_train = load_npz(FEATURE_DIR / "X_train.npz")
    X_val = load_npz(FEATURE_DIR / "X_validation.npz")
    X_test = load_npz(FEATURE_DIR / "X_test.npz")

    y_train = np.load(FEATURE_DIR / "y_train.npy")
    y_val = np.load(FEATURE_DIR / "y_validation.npy")
    y_test = np.load(FEATURE_DIR / "y_test.npy")

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    print("Loading feature datasets...")

    X_train, X_val, X_test, y_train, y_val, y_test = load_features()

    print("Training shape:", X_train.shape)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=30,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=30,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

        "SGD Classifier": SGDClassifier(
            loss="log_loss",
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = {}
    best_score = -1
    best_model = None
    best_name = None

    for name, model in models.items():

        print(f"\nTraining: {name}")

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1
        )

        print(
            "5-Fold CV F1:",
            round(cv_scores.mean(), 4),
            "+/-",
            round(cv_scores.std(), 4)
        )

        model.fit(X_train, y_train)

        val_predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, val_predictions)
        precision = precision_score(
            y_val, val_predictions,
            pos_label="FAKE",
            zero_division=0
        )
        recall = recall_score(
            y_val, val_predictions,
            pos_label="FAKE",
            zero_division=0
        )
        f1 = f1_score(
            y_val, val_predictions,
            pos_label="FAKE",
            zero_division=0
        )

        results[name] = {
            "cv_f1_macro_mean": float(cv_scores.mean()),
            "cv_f1_macro_std": float(cv_scores.std()),
            "validation_accuracy": float(accuracy),
            "validation_precision": float(precision),
            "validation_recall": float(recall),
            "validation_f1": float(f1)
        }

        print("Validation Accuracy:", round(accuracy, 4))
        print("Validation Precision:", round(precision, 4))
        print("Validation Recall:", round(recall, 4))
        print("Validation F1:", round(f1, 4))

        joblib.dump(
            model,
            MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib")
        )

        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name

    print("\nBest Model:", best_name)

    # Evaluate the selected model on test data only once
    test_predictions = best_model.predict(X_test)

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, test_predictions)),
        "precision_fake": float(precision_score(
            y_test, test_predictions,
            pos_label="FAKE",
            zero_division=0
        )),
        "recall_fake": float(recall_score(
            y_test, test_predictions,
            pos_label="FAKE",
            zero_division=0
        )),
        "f1_fake": float(f1_score(
            y_test, test_predictions,
            pos_label="FAKE",
            zero_division=0
        ))
    }

    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0
    )

    results["best_model"] = best_name
    results["test_metrics"] = test_metrics
    results["test_classification_report"] = report
    results["test_confusion_matrix"] = confusion_matrix(
        y_test, test_predictions
    ).tolist()

    with open(
        OUTPUT_DIR / "model_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    print("\nFINAL TEST RESULTS")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        test_predictions,
        zero_division=0
    ))

    print("\nModel training completed successfully!")


if __name__ == "__main__":
    main()
