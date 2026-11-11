"""Threshold selection and evaluation module for Feature A: Hobby Learning.

Conducts a chronological three-way split (70% train, 15% validation, 15% test)
based on order_id to experimentally determine the optimal classification threshold
for predicting exercise correctness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.hobby_learning.train import (
    DEFAULT_EVENTS_PATH,
    FEATURE_COLUMNS,
    ORDER_COLUMN,
    TARGET_COLUMN,
    load_processed_data,
    prepare_features_and_target,
)
from src.hobby_learning.train_hgb import build_hgb_pipeline
from src.hobby_learning.train_lr import build_logistic_regression_pipeline

THRESHOLDS: List[float] = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

REPORTS_DIR = Path("reports/hobby_learning/threshold_analysis")


def chronological_three_way_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    order_col: str = ORDER_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions into train, validation, and test partitions chronologically.

    Preserves chronological order using order_id to prevent temporal leakage.

    Args:
        df: Input interaction DataFrame.
        train_size: Fraction of data for training (default 0.70).
        val_size: Fraction of data for validation (default 0.15).
        test_size: Fraction of data for final testing (default 0.15).
        order_col: Column name representing chronological order.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(f"Split proportions must sum to 1.0, got {train_size + val_size + test_size}")

    sorted_df = df.sort_values(by=order_col).reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train_df = sorted_df.iloc[:train_end].reset_index(drop=True)
    val_df = sorted_df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = sorted_df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def classify_probabilities(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Classify probabilities into binary outcomes given a threshold.

    Decision rule: p_correct >= threshold -> 1, else 0.

    Args:
        probs: Array of predicted probabilities for positive class (1).
        threshold: Classification decision boundary.

    Returns:
        1D integer numpy array of binary predictions (0 or 1).
    """
    probs_arr = np.asarray(probs, dtype=float)
    return (probs_arr >= threshold).astype(int)


def compute_metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Calculate evaluation metrics for a specific classification threshold.

    Args:
        y_true: Ground truth binary targets.
        probs: Predicted probabilities for the positive class (1).
        threshold: Decision threshold.

    Returns:
        Dictionary containing threshold, accuracy, precision, recall, f1_score,
        specificity, balanced_accuracy, false_positives, false_negatives,
        true_positives, true_negatives, and confusion matrix list.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred = classify_probabilities(probs, threshold)

    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred))

    # Confusion matrix:
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": specificity,
        "balanced_accuracy": bal_acc,
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
        "true_positive_count": int(tp),
        "true_negative_count": int(tn),
        "confusion_matrix": cm.tolist(),
    }


def sweep_thresholds(
    y_true: pd.Series | np.ndarray,
    probs: np.ndarray,
    thresholds: List[float] = THRESHOLDS,
    model_name: str = "",
) -> List[Dict[str, Any]]:
    """Run threshold sweep over a range of threshold values."""
    results = []
    for th in thresholds:
        metrics = compute_metrics_at_threshold(y_true, probs, th)
        if model_name:
            metrics["model"] = model_name
        results.append(metrics)
    return results


def select_best_threshold(sweep_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select the best threshold based on primary metric: Validation F1-score.

    Tie-breaking: Prefer higher precision (to prevent overconfidence in student readiness).

    Returns:
        Dict detailing best_f1, highest_precision, highest_recall,
        highest_balanced_accuracy, and baseline_0_50.
    """
    sorted_by_f1 = sorted(
        sweep_results,
        key=lambda r: (round(r["f1_score"], 6), round(r["precision"], 6)),
        reverse=True,
    )
    best_f1_entry = sorted_by_f1[0]

    best_prec_entry = max(
        sweep_results,
        key=lambda r: (round(r["precision"], 6), round(r["f1_score"], 6)),
    )

    best_rec_entry = max(
        sweep_results,
        key=lambda r: (round(r["recall"], 6), round(r["f1_score"], 6)),
    )

    best_bal_acc_entry = max(
        sweep_results,
        key=lambda r: (round(r["balanced_accuracy"], 6), round(r["f1_score"], 6)),
    )

    baseline_0_50 = next((r for r in sweep_results if np.isclose(r["threshold"], 0.50)), None)

    return {
        "selected_best_f1": best_f1_entry,
        "highest_precision": best_prec_entry,
        "highest_recall": best_rec_entry,
        "highest_balanced_accuracy": best_bal_acc_entry,
        "baseline_0_50": baseline_0_50,
    }


def generate_threshold_plots(
    sweep_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot threshold vs Accuracy, Precision, Recall, and F1-score for evaluated models."""
    models = sweep_df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        sub_df = sweep_df[sweep_df["model"] == model_name].sort_values("threshold")
        ax.plot(sub_df["threshold"], sub_df["accuracy"], label="Accuracy", color="#2ca02c", marker="o", linewidth=1.5)
        ax.plot(sub_df["threshold"], sub_df["precision"], label="Precision", color="#1f77b4", marker="s", linewidth=1.5)
        ax.plot(sub_df["threshold"], sub_df["recall"], label="Recall", color="#d62728", marker="^", linewidth=1.5)
        ax.plot(sub_df["threshold"], sub_df["f1_score"], label="F1-Score", color="#9467bd", marker="D", linewidth=2.0)

        ax.axvline(0.50, color="gray", linestyle="--", alpha=0.7, label="Default (0.50)")
        ax.set_title(f"{model_name}\nMetrics vs. Threshold", fontsize=12, fontweight="bold")
        ax.set_xlabel("Classification Threshold", fontsize=11)
        ax.set_ylabel("Metric Score", fontsize=11)
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_confusion_matrix_plot(
    cm_selected_lr: np.ndarray,
    th_selected_lr: float,
    cm_050_lr: np.ndarray,
    cm_selected_hgb: np.ndarray,
    th_selected_hgb: float,
    cm_050_hgb: np.ndarray,
    output_path: Path,
) -> None:
    """Plot 2x2 comparison grid of confusion matrices on the test set."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    configs = [
        (axes[0, 0], cm_050_lr, "Logistic Regression @ 0.50 (Default)"),
        (axes[0, 1], cm_selected_lr, f"Logistic Regression @ {th_selected_lr:.2f} (Selected)"),
        (axes[1, 0], cm_050_hgb, "HistGradientBoosting @ 0.50 (Default)"),
        (axes[1, 1], cm_selected_hgb, f"HistGradientBoosting @ {th_selected_hgb:.2f} (Selected)"),
    ]

    for ax, cm, title in configs:
        ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(title, fontsize=11, fontweight="bold")
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Pred 0 (Incorrect)", "Pred 1 (Correct)"])
        ax.set_yticklabels(["True 0 (Incorrect)", "True 1 (Correct)"])

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:,}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if val > thresh else "black",
                    fontsize=12,
                    fontweight="bold",
                )
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_experiment(
    events_path: Path | str = DEFAULT_EVENTS_PATH,
    output_dir: Path | str = REPORTS_DIR,
) -> Dict[str, Any]:
    """Execute full threshold-selection experiment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading events dataset...")
    df = load_processed_data(events_path)

    print("Executing chronological three-way split (70% train, 15% val, 15% test)...")
    train_df, val_df, test_df = chronological_three_way_split(df)
    print(f"Dataset split counts: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    X_train, y_train = prepare_features_and_target(train_df, feature_cols=FEATURE_COLUMNS)
    X_val, y_val = prepare_features_and_target(val_df, feature_cols=FEATURE_COLUMNS)
    X_test, y_test = prepare_features_and_target(test_df, feature_cols=FEATURE_COLUMNS)

    # 1. Train models on Training partition only
    print("Training Logistic Regression pipeline...")
    lr_pipeline = build_logistic_regression_pipeline(feature_cols=FEATURE_COLUMNS, random_state=42)
    lr_pipeline.fit(X_train, y_train)

    print("Training HistGradientBoosting pipeline...")
    hgb_pipeline = build_hgb_pipeline(feature_cols=FEATURE_COLUMNS, random_state=42)
    hgb_pipeline.fit(X_train, y_train)

    # 2. Predict probabilities on Validation partition for threshold selection
    print("Computing validation probabilities...")
    val_probs_lr = lr_pipeline.predict_proba(X_val)[:, 1]
    val_probs_hgb = hgb_pipeline.predict_proba(X_val)[:, 1]

    print("Performing validation threshold sweeps...")
    sweep_lr = sweep_thresholds(y_val, val_probs_lr, thresholds=THRESHOLDS, model_name="LogisticRegression")
    sweep_hgb = sweep_thresholds(
        y_val, val_probs_hgb, thresholds=THRESHOLDS, model_name="HistGradientBoostingClassifier"
    )

    selection_lr = select_best_threshold(sweep_lr)
    selection_hgb = select_best_threshold(sweep_hgb)

    selected_th_lr = selection_lr["selected_best_f1"]["threshold"]
    selected_th_hgb = selection_hgb["selected_best_f1"]["threshold"]

    print(
        f"Logistic Regression: Selected threshold = {selected_th_lr:.2f} (Val F1: {selection_lr['selected_best_f1']['f1_score']:.4f})"
    )
    print(
        f"HistGradientBoosting: Selected threshold = {selected_th_hgb:.2f} (Val F1: {selection_hgb['selected_best_f1']['f1_score']:.4f})"
    )

    # 3. Evaluate selected thresholds on untouched Test partition
    print("Evaluating on untouched test partition...")
    test_probs_lr = lr_pipeline.predict_proba(X_test)[:, 1]
    test_probs_hgb = hgb_pipeline.predict_proba(X_test)[:, 1]

    test_metrics_lr_selected = compute_metrics_at_threshold(y_test, test_probs_lr, selected_th_lr)
    test_metrics_lr_selected["model"] = "LogisticRegression"
    test_metrics_lr_selected["threshold_type"] = "selected_best_f1"

    test_metrics_lr_050 = compute_metrics_at_threshold(y_test, test_probs_lr, 0.50)
    test_metrics_lr_050["model"] = "LogisticRegression"
    test_metrics_lr_050["threshold_type"] = "default_0.50"

    test_metrics_hgb_selected = compute_metrics_at_threshold(y_test, test_probs_hgb, selected_th_hgb)
    test_metrics_hgb_selected["model"] = "HistGradientBoostingClassifier"
    test_metrics_hgb_selected["threshold_type"] = "selected_best_f1"

    test_metrics_hgb_050 = compute_metrics_at_threshold(y_test, test_probs_hgb, 0.50)
    test_metrics_hgb_050["model"] = "HistGradientBoostingClassifier"
    test_metrics_hgb_050["threshold_type"] = "default_0.50"

    # 4. Generate CSV artifacts
    all_sweep = sweep_lr + sweep_hgb
    sweep_df = pd.DataFrame(all_sweep)
    sweep_csv_path = output_path / "threshold_comparison.csv"
    # Select clean ordering of columns
    cols_order = [
        "model",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "specificity",
        "balanced_accuracy",
        "false_positive_count",
        "false_negative_count",
        "true_positive_count",
        "true_negative_count",
    ]
    sweep_df[cols_order].to_csv(sweep_csv_path, index=False)
    print(f"Saved validation comparison to {sweep_csv_path}")

    test_comparisons = [
        test_metrics_lr_selected,
        test_metrics_lr_050,
        test_metrics_hgb_selected,
        test_metrics_hgb_050,
    ]
    test_df_metrics = pd.DataFrame(test_comparisons)
    test_csv_path = output_path / "test_set_comparison.csv"
    cols_test = ["model", "threshold_type"] + cols_order[1:]
    test_df_metrics[cols_test].to_csv(test_csv_path, index=False)
    print(f"Saved test comparison to {test_csv_path}")

    # 5. Generate plots
    plot_metrics_path = output_path / "threshold_vs_metrics.png"
    generate_threshold_plots(sweep_df, plot_metrics_path)
    print(f"Saved threshold metrics plot to {plot_metrics_path}")

    plot_cm_path = output_path / "confusion_matrices.png"
    generate_confusion_matrix_plot(
        cm_selected_lr=np.array(test_metrics_lr_selected["confusion_matrix"]),
        th_selected_lr=selected_th_lr,
        cm_050_lr=np.array(test_metrics_lr_050["confusion_matrix"]),
        cm_selected_hgb=np.array(test_metrics_hgb_selected["confusion_matrix"]),
        th_selected_hgb=selected_th_hgb,
        cm_050_hgb=np.array(test_metrics_hgb_050["confusion_matrix"]),
        output_path=plot_cm_path,
    )
    print(f"Saved confusion matrices plot to {plot_cm_path}")

    # 6. JSON output
    results_json = {
        "dataset": {
            "source": str(events_path),
            "total_records": len(df),
            "split_type": "chronological_order_id",
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "train_fraction": 0.70,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
        },
        "logistic_regression": {
            "validation_selection": selection_lr,
            "test_metrics_selected": test_metrics_lr_selected,
            "test_metrics_050": test_metrics_lr_050,
        },
        "hist_gradient_boosting": {
            "validation_selection": selection_hgb,
            "test_metrics_selected": test_metrics_hgb_selected,
            "test_metrics_050": test_metrics_hgb_050,
        },
    }

    json_path = output_path / "threshold_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved JSON results to {json_path}")

    return results_json


if __name__ == "__main__":
    run_experiment()
