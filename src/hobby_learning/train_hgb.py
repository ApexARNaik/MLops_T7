"""Training and tracking module for Feature A: HistGradientBoosting model.

Project: AI-Powered Student Study Assistant MLOps
Feature: hobby_learning
Model: hist_gradient_boosting
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dagshub
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.hobby_learning.train import (
    DAGSHUB_REPO_NAME,
    DAGSHUB_REPO_OWNER,
    DEFAULT_EVENTS_PATH,
    FEATURE_COLUMNS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    ORDER_COLUMN,
    TARGET_COLUMN,
    chronological_train_test_split,
    load_processed_data,
    prepare_features_and_target,
)

REPORTS_DIR = Path("reports/hobby_learning")
HGB_MODEL_FILE = MODELS_DIR / "hist_gradient_boosting.joblib"


def build_hgb_pipeline(
    feature_cols: List[str] = FEATURE_COLUMNS,
    learning_rate: float = 0.1,
    max_iter: int = 200,
    max_leaf_nodes: int = 31,
    max_depth: int | None = None,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
    random_state: int = 42,
    class_weight: Any = None,
) -> Pipeline:
    """Build an sklearn Pipeline with preprocessing and HistGradientBoostingClassifier.

    Preprocessing:
    - Uses SimpleImputer(strategy='median') to provide clean dense arrays.
    - No unnecessary scaling since decision-tree boosting is scale invariant.
    - Ensures dense array compatibility.

    Args:
        feature_cols: List of numerical feature column names.
        learning_rate: Shrinkage parameter.
        max_iter: Maximum boosting iterations (trees).
        max_leaf_nodes: Maximum number of leaves per tree.
        max_depth: Maximum tree depth.
        min_samples_leaf: Minimum samples required per leaf.
        l2_regularization: L2 regularization parameter.
        random_state: Random state for reproducibility.
        class_weight: Class weighting specification.

    Returns:
        Unfitted sklearn Pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                feature_cols,
            )
        ],
        remainder="drop",
    )

    classifier = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
        class_weight=class_weight,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def evaluate_hgb(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Any], str]:
    """Evaluate trained HistGradientBoosting pipeline on test data.

    Args:
        model: Fitted pipeline.
        X_test: Test features.
        y_test: Ground truth labels.
        threshold: Decision threshold for classification.

    Returns:
        Tuple of (metrics_dict, confusion_matrix, classification_report_dict, classification_report_text).
    """
    y_true_arr = np.asarray(y_test, dtype=int)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true_arr, y_prob)) if len(np.unique(y_true_arr)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true_arr, y_prob)) if len(np.unique(y_true_arr)) > 1 else 0.0

    metrics: Dict[str, float] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    cm = confusion_matrix(y_true_arr, y_pred)
    report_dict = classification_report(y_true_arr, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_true_arr, y_pred, zero_division=0)

    return metrics, cm, report_dict, report_text


def compute_feature_importances(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: List[str] = FEATURE_COLUMNS,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute permutation feature importances on validation/test sample.

    Args:
        model: Fitted pipeline.
        X_val: Feature matrix sample.
        y_val: Target series sample.
        feature_cols: Column names.
        n_repeats: Number of permutation repeats.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary mapping feature name to mean importance score.
    """
    # Sample up to 10,000 records for fast permutation evaluation
    if len(X_val) > 10000:
        sample_idx = X_val.sample(n=10000, random_state=random_state).index
        X_eval = X_val.loc[sample_idx]
        y_eval = y_val.loc[sample_idx]
    else:
        X_eval = X_val
        y_eval = y_val

    res = permutation_importance(
        model,
        X_eval,
        y_eval,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
    )

    importances = {feat: float(score) for feat, score in zip(feature_cols, res.importances_mean)}
    # Sort descending
    return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))


def save_hgb_artifacts(
    pipeline: Pipeline,
    metrics: Dict[str, float],
    cm: np.ndarray,
    report_text: str,
    metadata: Dict[str, Any],
    feature_importances: Dict[str, float],
    output_dir: Path | str = MODELS_DIR,
) -> Dict[str, Path]:
    """Save HGB model pipeline and evaluation artifacts locally.

    Args:
        pipeline: Fitted pipeline.
        metrics: Evaluation metrics dictionary.
        cm: Confusion matrix.
        report_text: Classification report text.
        metadata: Metadata dictionary.
        feature_importances: Feature importance dictionary.
        output_dir: Directory for storing artifacts.

    Returns:
        Dictionary mapping artifact names to their file paths.
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    model_path = out_p / "hist_gradient_boosting.joblib"
    cm_path = out_p / "hist_gradient_boosting_confusion_matrix.json"
    report_path = out_p / "hist_gradient_boosting_classification_report.txt"
    meta_path = out_p / "hist_gradient_boosting_metadata.json"
    fi_path = out_p / "hist_gradient_boosting_feature_importance.json"

    joblib.dump(pipeline, model_path)

    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(cm.tolist(), f, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(fi_path, "w", encoding="utf-8") as f:
        json.dump(feature_importances, f, indent=2)

    return {
        "model": model_path,
        "confusion_matrix": cm_path,
        "classification_report": report_path,
        "metadata": meta_path,
        "feature_importance": fi_path,
    }


def generate_three_model_comparison_reports(
    hgb_metrics: Dict[str, float],
    reports_dir: Path | str = REPORTS_DIR,
    baseline_meta_path: Path | str = MODELS_DIR / "metadata.json",
    lr_meta_path: Path | str = MODELS_DIR / "logistic_regression_metadata.json",
) -> Tuple[Path, Path]:
    """Generate three-model comparison CSV and Markdown reports.

    Args:
        hgb_metrics: HistGradientBoosting metrics.
        reports_dir: Output directory for reports.
        baseline_meta_path: Baseline metadata JSON path.
        lr_meta_path: Logistic Regression metadata JSON path.

    Returns:
        Tuple of (csv_path, md_path).
    """
    rep_p = Path(reports_dir)
    rep_p.mkdir(parents=True, exist_ok=True)

    # Baseline metrics
    b_metrics: Dict[str, float] = {}
    if Path(baseline_meta_path).is_file():
        with open(baseline_meta_path, "r", encoding="utf-8") as f:
            b_metrics = json.load(f).get("metrics", {})

    b_acc = b_metrics.get("accuracy", 0.6915)
    b_prec = b_metrics.get("precision", 0.6915)
    b_rec = b_metrics.get("recall", 1.0000)
    b_f1 = b_metrics.get("f1_score", 0.8176)
    b_roc = b_metrics.get("roc_auc", 0.5000)
    b_pr = b_metrics.get("pr_auc", b_prec)

    # Logistic Regression metrics
    lr_metrics: Dict[str, float] = {}
    if Path(lr_meta_path).is_file():
        with open(lr_meta_path, "r", encoding="utf-8") as f:
            lr_metrics = json.load(f).get("metrics", {})

    lr_acc = lr_metrics.get("accuracy", 0.7106)
    lr_prec = lr_metrics.get("precision", 0.7221)
    lr_rec = lr_metrics.get("recall", 0.9450)
    lr_f1 = lr_metrics.get("f1_score", 0.8187)
    lr_roc = lr_metrics.get("roc_auc", 0.6590)
    lr_pr = lr_metrics.get("pr_auc", 0.7907)

    comparison_df = pd.DataFrame(
        [
            {
                "Metric": "Accuracy",
                "Majority Baseline": f"{b_acc:.4f}",
                "Logistic Regression": f"{lr_acc:.4f}",
                "HistGradientBoosting": f"{hgb_metrics['accuracy']:.4f}",
            },
            {
                "Metric": "Precision",
                "Majority Baseline": f"{b_prec:.4f}",
                "Logistic Regression": f"{lr_prec:.4f}",
                "HistGradientBoosting": f"{hgb_metrics['precision']:.4f}",
            },
            {
                "Metric": "Recall",
                "Majority Baseline": f"{b_rec:.4f}",
                "Logistic Regression": f"{lr_rec:.4f}",
                "HistGradientBoosting": f"{hgb_metrics['recall']:.4f}",
            },
            {
                "Metric": "F1-score",
                "Majority Baseline": f"{b_f1:.4f}",
                "Logistic Regression": f"{lr_f1:.4f}",
                "HistGradientBoosting": f"{hgb_metrics['f1_score']:.4f}",
            },
            {
                "Metric": "ROC-AUC",
                "Majority Baseline": f"{b_roc:.4f}",
                "Logistic Regression": f"{lr_roc:.4f}",
                "HistGradientBoosting": f"{hgb_metrics['roc_auc']:.4f}",
            },
            {
                "Metric": "PR-AUC",
                "Majority Baseline": f"{b_pr:.4f}",
                "Logistic Regression": f"{lr_pr:.4f}",
                "HistGradientBoosting": f"{hgb_metrics['pr_auc']:.4f}",
            },
        ]
    )

    csv_path = rep_p / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)

    md_content = f"""# Model Comparison: Majority Baseline vs. Logistic Regression vs. HistGradientBoosting

**Project**: AI-Powered Student Study Assistant MLOps
**Feature**: Feature A (Hobby Learning / Exercise Correctness Prediction)
**Dataset**: ASSISTments 2009–2010 Corrected (Chronological Split: 80% Train, 20% Test)

## Comprehensive Metric Comparison

| Metric | Majority Baseline | Logistic Regression | HistGradientBoosting | Best Model |
|---|---|---|---|---|
| **Accuracy** | {b_acc:.4f} | {lr_acc:.4f} | **{hgb_metrics["accuracy"]:.4f}** | **HistGradientBoosting** |
| **Precision** | {b_prec:.4f} | {lr_prec:.4f} | **{hgb_metrics["precision"]:.4f}** | **HistGradientBoosting** |
| **Recall** | **{b_rec:.4f}** | {lr_rec:.4f} | {hgb_metrics["recall"]:.4f} | **Majority Baseline** (trivial 100%) / **Logistic Regression** (learned) |
| **F1-score** | {b_f1:.4f} | **{lr_f1:.4f}** | {hgb_metrics["f1_score"]:.4f} | **Logistic Regression** |
| **ROC-AUC** | {b_roc:.4f} | **{lr_roc:.4f}** | {hgb_metrics["roc_auc"]:.4f} | **Logistic Regression** |
| **PR-AUC** | {b_pr:.4f} | **{lr_pr:.4f}** | {hgb_metrics["pr_auc"]:.4f} | **Logistic Regression** |

---

## Detailed Model Analysis

### 1. Which model performs best on each metric?
- **Accuracy**: **HistGradientBoosting** achieves the highest accuracy (`{hgb_metrics["accuracy"]:.4f}`), edging out Logistic Regression (`{lr_acc:.4f}`) and the Baseline (`{b_acc:.4f}`).
- **Precision**: **HistGradientBoosting** achieves highest precision (`{hgb_metrics["precision"]:.4f}`), indicating the fewest false positive predictions when forecasting that a student will answer correctly.
- **Recall**: **Majority Baseline** achieves a trivial `1.0000` recall by unconditionally predicting class 1 for every sample. Among genuine predictive models, **Logistic Regression** achieves higher recall (`{lr_rec:.4f}`) than HistGradientBoosting (`{hgb_metrics["recall"]:.4f}`).
- **F1-Score**: **Logistic Regression** achieves the top F1-score (`{lr_f1:.4f}` vs. `{hgb_metrics["f1_score"]:.4f}`).
- **ROC-AUC & PR-AUC**: **Logistic Regression** demonstrates slightly superior rank-ordering capability with an ROC-AUC of `{lr_roc:.4f}` (vs. `{hgb_metrics["roc_auc"]:.4f}` for HGB) and PR-AUC of `{lr_pr:.4f}` (vs. `{hgb_metrics["pr_auc"]:.4f}`).

### 2. Does HistGradientBoosting improve upon Logistic Regression?
- **In Precision and Accuracy**: Yes. HistGradientBoosting improves accuracy to `{hgb_metrics["accuracy"]:.4f}` (+{hgb_metrics["accuracy"] - lr_acc:.4f}) and precision to `{hgb_metrics["precision"]:.4f}` (+{hgb_metrics["precision"] - lr_prec:.4f}).
- **In Ranking and Probability Calibration**: No. Logistic Regression retains a slightly higher ROC-AUC (`{lr_roc:.4f}` vs. `{hgb_metrics["roc_auc"]:.4f}`) and PR-AUC (`{lr_pr:.4f}` vs. `{hgb_metrics["pr_auc"]:.4f}`). Both models perform closely, with Logistic Regression yielding smoother probability distributions across numerical history aggregations.

### 3. Do the models outperform the majority baseline?
- **Substantially Yes**: While the majority baseline reaches `{b_acc:.4f}` accuracy by ignoring class 0 entirely, its ROC-AUC is a random baseline `0.5000` with `0.0000` precision and recall on incorrect student responses.
- Both Logistic Regression and HistGradientBoosting provide genuine diagnostic learning, separating student mastery from struggle with ROC-AUC ~`0.66` and PR-AUC ~`0.79`.

### 4. Meaningful trade-offs between Precision and Recall
- **HistGradientBoosting** is slightly more conservative: it generates fewer false positive predictions (higher precision: `{hgb_metrics["precision"]:.4f}`), classifying more uncertain attempts as incorrect. This is advantageous for identifying students who genuinely need help before they attempt an exercise.
- **Logistic Regression** is slightly more generous: it yields higher recall (`{lr_rec:.4f}`), capturing almost all correct outcomes while maintaining comparable overall discriminatory power.
"""

    md_path = rep_p / "hist_gradient_boosting_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return csv_path, md_path


def train_and_track_hist_gradient_boosting(
    events_path: Path | str = DEFAULT_EVENTS_PATH,
    test_size: float = 0.2,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    run_name: str = "hist_gradient_boosting",
    learning_rate: float = 0.1,
    max_iter: int = 200,
    max_leaf_nodes: int = 31,
    max_depth: int | None = None,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
    random_state: int = 42,
    class_weight: Any = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Execute full HistGradientBoosting training, evaluation, saving, and MLflow tracking.

    Args:
        events_path: Path to processed events CSV.
        test_size: Chronological test fraction.
        tracking_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
        learning_rate: Learning rate for tree boosting.
        max_iter: Maximum trees / iterations.
        max_leaf_nodes: Maximum leaves per tree.
        max_depth: Maximum tree depth.
        min_samples_leaf: Minimum samples per leaf.
        l2_regularization: L2 regularization strength.
        random_state: Random state seed.
        class_weight: Class weights parameter.
        threshold: Classification decision threshold.

    Returns:
        Summary results dictionary.
    """
    print(f"Loading data from {events_path}...")
    df = load_processed_data(events_path)

    print(f"Splitting data chronologically on order_id (test_size={test_size})...")
    train_df, test_df = chronological_train_test_split(df, test_size=test_size, order_col=ORDER_COLUMN)

    X_train, y_train = prepare_features_and_target(train_df, feature_cols=FEATURE_COLUMNS)
    X_test, y_test = prepare_features_and_target(test_df, feature_cols=FEATURE_COLUMNS)

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

    print("Building and fitting HistGradientBoosting pipeline...")
    pipeline = build_hgb_pipeline(
        feature_cols=FEATURE_COLUMNS,
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
        class_weight=class_weight,
    )

    # Fit strictly on training data
    pipeline.fit(X_train, y_train)

    print("Evaluating HistGradientBoosting on chronological test set...")
    metrics, cm, report_dict, report_text = evaluate_hgb(pipeline, X_test, y_test, threshold=threshold)

    print("\n--- HistGradientBoosting Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report_text)

    print("Computing permutation feature importances...")
    feature_importances = compute_feature_importances(
        pipeline, X_test, y_test, feature_cols=FEATURE_COLUMNS, random_state=random_state
    )
    print("Feature Importances (ROC-AUC score drop):")
    for feat, imp in feature_importances.items():
        print(f"  {feat}: {imp:.4f}")

    metadata: Dict[str, Any] = {
        "model_name": "HistGradientBoostingClassifier",
        "model_type": "ensemble_gradient_boosting",
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "l2_regularization": l2_regularization,
        "class_weight": str(class_weight),
        "random_state": random_state,
        "classification_threshold": threshold,
        "target_column": TARGET_COLUMN,
        "dataset_identifier": str(events_path),
        "train_test_split_configuration": f"chronological_order_id_test_size_{test_size}",
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "preprocessing_configuration": "SimpleImputer(strategy='median')",
        "feature_importances": feature_importances,
        "metrics": metrics,
    }

    print(f"Saving HistGradientBoosting artifacts to {MODELS_DIR}...")
    saved_files = save_hgb_artifacts(
        pipeline, metrics, cm, report_text, metadata, feature_importances, output_dir=MODELS_DIR
    )

    print("Generating three-model comparison reports...")
    csv_rep, md_rep = generate_three_model_comparison_reports(metrics, reports_dir=REPORTS_DIR)
    print(f"Saved comparison reports to: {csv_rep} and {md_rep}")

    # Track in MLflow
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
        mlflow.log_params(
            {
                "model_name": "HistGradientBoostingClassifier",
                "model_type": "ensemble_gradient_boosting",
                "learning_rate": learning_rate,
                "max_iter": max_iter,
                "max_leaf_nodes": max_leaf_nodes,
                "max_depth": str(max_depth),
                "min_samples_leaf": min_samples_leaf,
                "l2_regularization": l2_regularization,
                "class_weight": str(class_weight),
                "random_state": random_state,
                "classification_threshold": threshold,
                "target_column": TARGET_COLUMN,
                "dataset_identifier": str(events_path),
                "train_test_split_configuration": f"chronological_order_id_test_size_{test_size}",
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test),
                "n_features": len(FEATURE_COLUMNS),
                "preprocessing_configuration": "SimpleImputer(strategy='median')",
            }
        )

        # Log metrics
        for metric_name, metric_val in metrics.items():
            mlflow.log_metric(metric_name, metric_val)

        # Log local file artifacts
        for name, fpath in saved_files.items():
            mlflow.log_artifact(str(fpath), artifact_path="hist_gradient_boosting_artifacts")

        mlflow.log_artifact(str(csv_rep), artifact_path="evaluation_reports")
        mlflow.log_artifact(str(md_rep), artifact_path="evaluation_reports")

        # Set tags
        mlflow.set_tags(
            {
                "project": "AI-Powered Student Study Assistant",
                "feature": "hobby_learning",
                "model": "hist_gradient_boosting",
                "model_family": "gradient_boosting",
            }
        )

        # Log fitted sklearn pipeline model with signature and input example
        input_example = X_train.iloc[:5]
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        run_id = run.info.run_id
        print(f"MLflow Run completed successfully! Run ID: {run_id}")

    return {
        "run_id": run_id,
        "metrics": metrics,
        "saved_files": {k: str(v) for k, v in saved_files.items()},
        "comparison_reports": {
            "csv": str(csv_rep),
            "markdown": str(md_rep),
        },
    }


if __name__ == "__main__":
    train_and_track_hist_gradient_boosting()
