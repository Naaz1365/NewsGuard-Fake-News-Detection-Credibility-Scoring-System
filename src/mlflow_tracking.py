import json
import math
import re
from pathlib import Path

import mlflow

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_FILE = BASE_DIR / "outputs" / "model_results.json"

# SQLite tracking backend
mlflow.set_tracking_uri("./mlruns")

# Load existing model results (no retraining required)
with open(RESULTS_FILE, "r", encoding="utf-8") as file:
    results = json.load(file)

# Create or select experiment
mlflow.set_experiment("NewsGuard-Fake-News-Detection")


def log_numeric_metrics(data, prefix=""):
    """Log numeric metrics from nested dictionaries."""
    if isinstance(data, dict):
        for key, value in data.items():
            name = f"{prefix}_{key}" if prefix else key
            log_numeric_metrics(value, name)

    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        if math.isfinite(float(data)):
            metric_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", prefix)[:250]
            mlflow.log_metric(metric_name, float(data))


with mlflow.start_run(run_name="ExtraTrees-Fake-News-Model"):

    # Parameters
    mlflow.log_param("best_model", results.get("best_model", "Extra Trees"))
    mlflow.log_param("dataset", "LIAR")
    mlflow.log_param(
    "feature_extraction",
    "TF-IDF + Word2Vec + Readability + Sentiment"
)
    mlflow.log_param("tfidf_max_features", 10000)
    mlflow.log_param("word2vec_dimensions", 100)
    mlflow.log_param("auxiliary_features", "Flesch Readability + VADER Sentiment")
    mlflow.log_param("total_features", 9983)
    mlflow.log_param("cross_validation", "Stratified 5-Fold")
    mlflow.log_param("train_samples", 6434)
    mlflow.log_param("validation_samples", 804)
    mlflow.log_param("test_samples", 805)

    # Log all available numeric results
    log_numeric_metrics(results)

    # Save original results JSON as an artifact
    mlflow.log_artifact(str(RESULTS_FILE))

    # Save feature importance outputs if available
    importance_file = BASE_DIR / "outputs" / "feature_importance.csv"
    if importance_file.exists():
        mlflow.log_artifact(str(importance_file))

    plot_file = BASE_DIR / "outputs" / "plots" / "feature_importance.png"
    if plot_file.exists():
        mlflow.log_artifact(str(plot_file))

    print("MLflow experiment logged successfully!")
    print("Experiment: NewsGuard-Fake-News-Detection")
    print("Best model:", results.get("best_model", "Extra Trees"))
    print("Test metrics:", results.get("test_metrics", {}))
    print("Run ID:", mlflow.active_run().info.run_id)