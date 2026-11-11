"""Training pipeline for Feature A: Hobby Learning - Majority Baseline Model.

Project: AI-Powered Student Study Assistant MLOps
Owner: Teammate AT
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dagshub
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Configuration constants
DEFAULT_EVENTS_PATH = Path("data/hobby_learning/processed/events.csv")
MODELS_DIR = Path("models/hobby_learning")
MLFLOW_TRACKING_URI = "https://dagshub.com/ApexARNaik/MLops_T7.mlflow"
MLFLOW_EXPERIMENT_NAME = "hobby_learning"
RUN_NAME = "majority_baseline"
DAGSHUB_REPO_OWNER = "ApexARNaik"
DAGSHUB_REPO_NAME = "MLops_T7"

# Feature definitions strictly preventing current-row outcome leakage
TARGET_COLUMN = "correct"
ORDER_COLUMN = "order_id"

FEATURE_COLUMNS = [
    "user_prev_count",
    "user_prev_accuracy",
    "user_prev_attempts_mean",
    "user_prev_hints_mean",
    "user_skill_prev_count",
    "user_skill_prev_accuracy",
    "n_skill_components",
]


def load_processed_data(file_path: Path | str = DEFAULT_EVENTS_PATH) -> pd.DataFrame:
    """Load the processed events dataset.

    Args:
        file_path: Path to events.csv.

    Returns:
        pd.DataFrame containing processed interaction events.

    Raises:
        FileNotFoundError: If the processed events file is not found.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed events file not found at: {path}. "
            f"Please run `python -m src.hobby_learning.data` first."
        )
    return pd.read_csv(path)


def prepare_features_and_target(
    df: pd.DataFrame,
    feature_cols: List[str] | None = None,
    target_col: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract features and target while ensuring no target leakage.

    Args:
        df: Input DataFrame.
        feature_cols: List of column names to use as features.
        target_col: Target column name.

    Returns:
        Tuple of (X DataFrame, y Series).

    Raises:
        ValueError: If target column is included in feature columns.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    # Prevent target or current-attempt leakage
    forbidden = {target_col, "attempt_count", "hint_count", "ms_first_response", "score"}
    leaked = set(feature_cols).intersection(forbidden)
    if leaked:
        raise ValueError(f"Data leakage detected! Forbidden columns in features: {leaked}")

    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()
    return X, y


def chronological_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    order_col: str = ORDER_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions into train and test sets strictly chronologically.

    Preserves chronological order using numeric order_id to prevent
    future-to-past data leakage.

    Args:
        df: Input DataFrame.
        test_size: Proportion of dataset to include in the test split.
        order_col: Column to sort by chronologically.

    Returns:
        Tuple of (train_df, test_df).
    """
    sorted_df = df.sort_values(by=order_col).reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1.0 - test_size))
    train_df = sorted_df.iloc[:split_idx].reset_index(drop=True)
    test_df = sorted_df.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df


def train_majority_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> DummyClassifier:
    """Train a DummyClassifier using the majority-class strategy.

    Args:
        X_train: Training feature matrix.
        y_train: Training binary targets.

    Returns:
        Fitted DummyClassifier.
    """
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_train, y_train)
    return clf


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Any], str]:
    """Compute classification metrics, confusion matrix, and report.

    Args:
        y_true: True binary targets.
        y_pred: Predicted binary labels.
        y_prob: Predicted positive class probabilities (optional).

    Returns:
        Tuple of (metrics_dict, confusion_matrix, classification_report_dict, classification_report_text).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    metrics: Dict[str, float] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
    }

    if y_prob is not None:
        try:
            # Check if there are at least two distinct classes in true labels
            if len(np.unique(y_true_arr)) > 1:
                auc = float(roc_auc_score(y_true_arr, y_prob))
                metrics["roc_auc"] = auc
        except Exception:
            pass

    cm = confusion_matrix(y_true_arr, y_pred)
    report_dict = classification_report(y_true_arr, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_true_arr, y_pred, zero_division=0)

    return metrics, cm, report_dict, report_text


def save_baseline_artifacts(
    model: DummyClassifier,
    metrics: Dict[str, float],
    cm: np.ndarray,
    report_text: str,
    metadata: Dict[str, Any],
    output_dir: Path | str = MODELS_DIR,
) -> Dict[str, Path]:
    """Save model and evaluation artifacts locally.

    Args:
        model: Trained DummyClassifier model.
        metrics: Evaluation metrics dictionary.
        cm: Confusion matrix numpy array.
        report_text: Classification report text.
        metadata: Training run metadata.
        output_dir: Target directory for saving.

    Returns:
        Dictionary mapping artifact names to their file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_file = out_path / "baseline_model.joblib"
    cm_file = out_path / "confusion_matrix.json"
    report_file = out_path / "classification_report.txt"
    meta_file = out_path / "metadata.json"

    joblib.dump(model, model_file)

    with open(cm_file, "w", encoding="utf-8") as f:
        json.dump(cm.tolist(), f, indent=2)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "model": model_file,
        "confusion_matrix": cm_file,
        "classification_report": report_file,
        "metadata": meta_file,
    }


def train_and_track(
    events_path: Path | str = DEFAULT_EVENTS_PATH,
    test_size: float = 0.2,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    run_name: str = RUN_NAME,
) -> Dict[str, Any]:
    """Execute training pipeline and track run in MLflow / DagsHub.

    Args:
        events_path: Path to processed events.csv.
        test_size: Chronological test set fraction.
        tracking_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.

    Returns:
        Summary dictionary of results and metrics.
    """
    print(f"Loading data from {events_path}...")
    df = load_processed_data(events_path)

    print(f"Splitting data chronologically (test_size={test_size})...")
    train_df, test_df = chronological_train_test_split(df, test_size=test_size)

    X_train, y_train = prepare_features_and_target(train_df)
    X_test, y_test = prepare_features_and_target(test_df)

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Train majority class: {y_train.mode()[0]}")

    print("Training DummyClassifier (strategy='most_frequent')...")
    clf = train_majority_baseline(X_train, y_train)

    print("Evaluating predictions on chronological test set...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    metrics, cm, report_dict, report_text = evaluate_predictions(y_test, y_pred, y_prob)

    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report_text)

    metadata: Dict[str, Any] = {
        "model_name": "DummyClassifier",
        "strategy": "most_frequent",
        "target_column": TARGET_COLUMN,
        "training_dataset": str(events_path),
        "train_test_split_configuration": f"chronological_order_id_test_size_{test_size}",
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
    }

    # Save artifacts locally
    print(f"Saving baseline artifacts to {MODELS_DIR}...")
    saved_files = save_baseline_artifacts(clf, metrics, cm, report_text, metadata, output_dir=MODELS_DIR)

    # MLflow Tracking
    print(f"Tracking run in MLflow (URI: {tracking_uri}, Experiment: {experiment_name})...")
    try:
        dagshub.init(
            repo_owner=DAGSHUB_REPO_OWNER,
            repo_name=DAGSHUB_REPO_NAME,
            mlflow=True,
        )
    except Exception as e:
        print(f"Note: dagshub.init encountered: {e}. Falling back to mlflow URI directly.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_params({
            "model_name": "DummyClassifier",
            "model_strategy": "most_frequent",
            "target_column": TARGET_COLUMN,
            "training_dataset": str(events_path),
            "train_test_split_configuration": f"chronological_split_{test_size}",
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": len(FEATURE_COLUMNS),
        })

        # Log metrics
        for metric_name, metric_val in metrics.items():
            mlflow.log_metric(metric_name, metric_val)

        # Log artifacts
        for name, fpath in saved_files.items():
            mlflow.log_artifact(str(fpath), artifact_path="baseline_artifacts")

        # Set tags
        mlflow.set_tags({
            "project": "AI-Powered Student Study Assistant MLOps",
            "feature": "Feature A: Hobby Learning",
            "model_type": "baseline",
            "model_family": "heuristic",
        })

        run_id = run.info.run_id
        print(f"MLflow Run completed successfully! Run ID: {run_id}")

    return {
        "run_id": run_id,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "metadata": metadata,
        "saved_files": {k: str(v) for k, v in saved_files.items()},
    }


if __name__ == "__main__":
    train_and_track()

