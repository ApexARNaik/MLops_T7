"""
Training and Evaluation Pipeline for Answer Feedback Module
Owner: Member 2 (AV)
Project: AI-Powered Student Study Assistant MLOps Project
Path: src/answer_feedback/train.py

Implements:
- Baseline 0: Per-Question Mean baseline
- Leakage-free GroupKFold (5 splits grouped by student_id)
- Evaluation metrics: MAE, RMSE, Exact mark accuracy, Within-0.5 accuracy, Within-1.0 accuracy
- Mean and standard deviation across folds
- MLflow tracking (sqlite:///mlflow_B.db, experiment 'answer_feedback')
- Metrics report CSV generation in docs/ (and reports/ if available)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.answer_feedback.data import build_long_format_marks, PROCESSED_LONG_PATH

MLFLOW_TRACKING_URI = "sqlite:///mlflow_B.db"
MLFLOW_EXPERIMENT = "answer_feedback"
RANDOM_SEED = 42
N_FOLDS = 5


class QuestionMeanBaseline:
    """
    Per-Question Mean baseline model.
    Learns question-level mean marks strictly from training student responses.
    Provides fallback to global mean for unseen questions.
    """
    def __init__(self, step: float = 0.5, max_mark: float = 2.0):
        self.step = step
        self.max_mark = max_mark
        self.question_means_ = {}
        self.global_mean_ = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits question means on training fold data."""
        df = X.copy()
        df["target"] = y.values
        self.global_mean_ = float(df["target"].mean())
        self.question_means_ = df.groupby("qid")["target"].mean().to_dict()
        return self

    def predict(self, X: pd.DataFrame, round_to_step: bool = True) -> np.ndarray:
        """
        Predicts mark for each row based on question ID.
        Clips to [0, max_mark] and optionally rounds to allowed step (0.5).
        """
        raw_preds = np.array([self.question_means_.get(q, self.global_mean_) for q in X["qid"]])
        clipped = np.clip(raw_preds, 0.0, self.max_mark)
        if round_to_step:
            factor = 1.0 / self.step
            return np.clip(np.round(clipped * factor) / factor, 0.0, self.max_mark)
        return clipped


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates all 5 required evaluation metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    exact_acc = float(np.mean(np.isclose(y_true, y_pred, atol=1e-5)))
    within_05_acc = float(np.mean(np.abs(y_true - y_pred) <= 0.5 + 1e-5))
    within_10_acc = float(np.mean(np.abs(y_true - y_pred) <= 1.0 + 1e-5))
    return {
        "mae": mae,
        "rmse": rmse,
        "exact_accuracy": exact_acc,
        "within_0.5_accuracy": within_05_acc,
        "within_1.0_accuracy": within_10_acc,
    }


def run_baseline_cv():
    print("=" * 80)
    print("ANSWER FEEDBACK MODULE: BASELINE EVALUATION (PER-QUESTION MEAN)")
    print("=" * 80)

    # 1. Load or build long-format dataset
    if os.path.exists(PROCESSED_LONG_PATH):
        df = pd.read_csv(PROCESSED_LONG_PATH)
    else:
        df = build_long_format_marks()

    n_samples = len(df)
    n_students = df["student_id"].nunique()
    print(f"Loaded dataset: {n_samples} response rows across {n_students} distinct students.")
    print(f"Target distribution summary: mean={df['mark'].mean():.2f}, min={df['mark'].min()}, max={df['mark'].max()}")

    # 2. Setup GroupKFold cross-validation
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []
    oof_predictions = np.zeros(n_samples)

    print("\n--- Running 5-Fold GroupKFold Cross-Validation (Grouped by student_id) ---")
    print(f"{'Fold':<6} {'Train Stud':<12} {'Test Stud':<12} {'MAE':<8} {'RMSE':<8} {'Exact Acc':<11} {'<=0.5 Acc':<11} {'<=1.0 Acc':<11}")
    print("-" * 85)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["student_id"]), 1):
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]

        # Verify zero student leakage across train and test
        train_students = set(train_data["student_id"])
        test_students = set(test_data["student_id"])
        leakage = train_students.intersection(test_students)
        if leakage:
            raise ValueError(f"[FATAL] Data leakage detected in fold {fold}: {leakage}")

        # Fit model on training fold only
        model = QuestionMeanBaseline(step=0.5, max_mark=2.0)
        model.fit(train_data[["qid"]], train_data["mark"])

        # Predict on test fold
        y_true = test_data["mark"].values
        y_pred = model.predict(test_data[["qid"]], round_to_step=True)
        oof_predictions[test_idx] = y_pred

        metrics = evaluate_predictions(y_true, y_pred)
        metrics["fold"] = fold
        metrics["n_train_samples"] = len(train_data)
        metrics["n_test_samples"] = len(test_data)
        metrics["n_train_students"] = len(train_students)
        metrics["n_test_students"] = len(test_students)
        fold_results.append(metrics)

        print(
            f"{fold:<6} {len(train_students):<12} {len(test_students):<12} "
            f"{metrics['mae']:<8.4f} {metrics['rmse']:<8.4f} "
            f"{metrics['exact_accuracy']:<11.4f} {metrics['within_0.5_accuracy']:<11.4f} "
            f"{metrics['within_1.0_accuracy']:<11.4f}"
        )

    # 3. Calculate mean and standard deviation across folds
    df_folds = pd.DataFrame(fold_results)
    metric_cols = ["mae", "rmse", "exact_accuracy", "within_0.5_accuracy", "within_1.0_accuracy"]
    means = df_folds[metric_cols].mean()
    stds = df_folds[metric_cols].std()

    print("-" * 85)
    print(f"{'MEAN':<32} {means['mae']:<8.4f} {means['rmse']:<8.4f} {means['exact_accuracy']:<11.4f} {means['within_0.5_accuracy']:<11.4f} {means['within_1.0_accuracy']:<11.4f}")
    print(f"{'STD':<32} {stds['mae']:<8.4f} {stds['rmse']:<8.4f} {stds['exact_accuracy']:<11.4f} {stds['within_0.5_accuracy']:<11.4f} {stds['within_1.0_accuracy']:<11.4f}")
    print("=" * 85)

    # 4. Out-of-fold per-question MAE analysis
    df["oof_pred"] = oof_predictions
    df["abs_error"] = np.abs(df["mark"] - df["oof_pred"])
    df["sq_error"] = (df["mark"] - df["oof_pred"]) ** 2
    df["exact_hit"] = np.isclose(df["mark"], df["oof_pred"], atol=1e-5)
    df["within_05_hit"] = df["abs_error"] <= 0.5 + 1e-5

    q_summary = df.groupby("qid").agg(
        n_samples=("mark", "count"),
        actual_mean=("mark", "mean"),
        oof_pred_mean=("oof_pred", "mean"),
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda x: np.sqrt(np.mean(x))),
        exact_accuracy=("exact_hit", "mean"),
        within_05_accuracy=("within_05_hit", "mean")
    ).reset_index().round(4)

    # 5. Save summary CSVs to docs/ (and reports/ if available)
    out_dir = "reports" if os.path.exists("reports") else "docs"
    os.makedirs(out_dir, exist_ok=True)

    summary_csv_path = os.path.join(out_dir, "answer_baseline_metrics.csv")
    summary_rows = []
    for r in fold_results:
        summary_rows.append({
            "fold": f"Fold_{r['fold']}",
            "mae": round(r["mae"], 4),
            "rmse": round(r["rmse"], 4),
            "exact_accuracy": round(r["exact_accuracy"], 4),
            "within_0.5_accuracy": round(r["within_0.5_accuracy"], 4),
            "within_1.0_accuracy": round(r["within_1.0_accuracy"], 4),
            "n_train_students": r["n_train_students"],
            "n_test_students": r["n_test_students"],
        })
    summary_rows.append({
        "fold": "Mean",
        "mae": round(means["mae"], 4),
        "rmse": round(means["rmse"], 4),
        "exact_accuracy": round(means["exact_accuracy"], 4),
        "within_0.5_accuracy": round(means["within_0.5_accuracy"], 4),
        "within_1.0_accuracy": round(means["within_1.0_accuracy"], 4),
        "n_train_students": 40,
        "n_test_students": 10,
    })
    summary_rows.append({
        "fold": "Std",
        "mae": round(stds["mae"], 4),
        "rmse": round(stds["rmse"], 4),
        "exact_accuracy": round(stds["exact_accuracy"], 4),
        "within_0.5_accuracy": round(stds["within_0.5_accuracy"], 4),
        "within_1.0_accuracy": round(stds["within_1.0_accuracy"], 4),
        "n_train_students": 0,
        "n_test_students": 0,
    })
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    print(f"\nSaved baseline metrics summary to: {summary_csv_path}")

    q_csv_path = os.path.join(out_dir, "per_question_baseline_mae.csv")
    q_summary.to_csv(q_csv_path, index=False)
    print(f"Saved per-question MAE breakdown to: {q_csv_path}")

    # 6. MLflow Tracking
    print("\n--- Logging Experiment to MLflow ---")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="baseline_question_mean") as run:
        # Log parameters
        mlflow.log_params({
            "model_name": "baseline_question_mean",
            "model_type": "QuestionMeanBaseline",
            "split_method": "GroupKFold(n_splits=5)",
            "group_column": "student_id",
            "random_seed": RANDOM_SEED,
            "n_folds": N_FOLDS,
            "n_students": n_students,
            "n_samples": n_samples,
            "questions": "Q21-Q35",
            "source_data": "Teacher_manual_marks_Anonymized.csv",
            "step_size": 0.5,
            "max_mark": 2.0
        })

        # Log aggregated metrics (mean and std)
        mlflow.log_metrics({
            "mae_mean": float(means["mae"]),
            "mae_std": float(stds["mae"]),
            "rmse_mean": float(means["rmse"]),
            "rmse_std": float(stds["rmse"]),
            "exact_accuracy_mean": float(means["exact_accuracy"]),
            "exact_accuracy_std": float(stds["exact_accuracy"]),
            "within_0.5_accuracy_mean": float(means["within_0.5_accuracy"]),
            "within_0.5_accuracy_std": float(stds["within_0.5_accuracy"]),
            "within_1.0_accuracy_mean": float(means["within_1.0_accuracy"]),
            "within_1.0_accuracy_std": float(stds["within_1.0_accuracy"]),
        })

        # Log per-fold metrics
        for r in fold_results:
            fold_num = r["fold"]
            for m in metric_cols:
                mlflow.log_metric(f"fold_{fold_num}_{m}", float(r[m]))

        # Log CSV artifacts
        mlflow.log_artifact(summary_csv_path, artifact_path="evaluation_reports")
        mlflow.log_artifact(q_csv_path, artifact_path="evaluation_reports")

        run_id = run.info.run_id
        print(f"MLflow Run logged successfully! Run ID: {run_id}")
        print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
        print(f"MLflow Experiment Name: {MLFLOW_EXPERIMENT}")

    return {
        "means": means.to_dict(),
        "stds": stds.to_dict(),
        "fold_results": fold_results,
        "run_id": run_id
    }


if __name__ == "__main__":
    run_baseline_cv()

