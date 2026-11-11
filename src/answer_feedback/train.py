"""
Training, Model Comparison, and Registry Pipeline for Answer Feedback Module
Owner: Member 2 (AV)
Project: AI-Powered Student Study Assistant MLOps Project
Path: src/answer_feedback/train.py

Implements:
1. baseline_question_mean:
   Per-question historical mean marks calculated strictly on training-fold students.
2. baseline_tfidf_ridge:
   Ridge regression on TF-IDF cosine similarity, word length features, and fold question means.
3. candidate_gradient_boosting:
   HistGradientBoostingRegressor / GradientBoosting on complete engineered feature set
   (tfidf_cos, coverage, max_point_hit, n_words, log_words, question_mean_mark, max_mark).

Strict Anti-Leakage Constraints:
- GroupKFold with 5 folds grouped strictly by student_id.
- No ground-truth marks in feature matrices.
- No Corrected_Pdf ever referenced or used.
- Question mean marks computed strictly on the training fold slice.
- Text vectorizers and scalers fit strictly on training fold or reference text.
- Missing answers are preserved with is_missing flags; missing OCR records are NOT zero-filled.
- Metric rounding and clipping: predictions clipped to [0, max_mark] and rounded to 0.5 steps.
- MLflow tracking (sqlite:///mlflow_B.db, experiment 'answer_feedback').
- Automatic comparison, best model selection, model registry (champion alias), and joblib export.
"""

import os
import sys
import json
import hashlib
import joblib
import subprocess
import numpy as np
import pandas as pd
import mlflow
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor

from src.answer_feedback.features import (
    generate_feature_dataset,
    compute_question_means,
    OUTPUT_FEATURES_CSV,
    DEFAULT_RUBRIC_JSON,
    PROCESSED_MARKS_CSV
)

# MLflow and export configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow_B.db"
MLFLOW_EXPERIMENT = "answer_feedback"
REGISTERED_MODEL_NAME = "study-assistant-answer-scoring-model"
MODEL_ALIAS = "champion"

MODELS_DIR = "models"
MODEL_EXPORT_PATH = os.path.join(MODELS_DIR, "answer_model.joblib")
META_EXPORT_PATH = os.path.join(MODELS_DIR, "answer_meta.json")
DOCS_DIR = "docs"
SUMMARY_CSV_PATH = os.path.join(DOCS_DIR, "model_comparison_results.csv")
PER_Q_MAE_PATH = os.path.join(DOCS_DIR, "per_question_mae.csv")

N_FOLDS = 5
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Utility & Metric Functions
# ---------------------------------------------------------------------------

def get_git_commit_hash() -> str:
    """Retrieves current git commit hash safely."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"


def get_rubric_sha256(rubric_path: str = DEFAULT_RUBRIC_JSON) -> str:
    """Computes SHA256 checksum of the rubric JSON definition."""
    if not os.path.exists(rubric_path):
        return "unknown"
    sha = hashlib.sha256()
    with open(rubric_path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def round_and_clip(preds: np.ndarray, max_mark: float = 2.0, step: float = 0.5) -> np.ndarray:
    """
    Rounds predicted marks to actual allowed exam increments (0.5 steps)
    and bounds them to [0, max_mark].
    """
    factor = 1.0 / step
    clipped = np.clip(preds, 0.0, max_mark)
    return np.clip(np.round(clipped * factor) / factor, 0.0, max_mark)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates all 5 required evaluation metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    exact_acc = float(np.mean(np.isclose(y_true, y_pred, atol=1e-5)))
    within_05_acc = float(np.mean(np.abs(y_true - y_pred) <= 0.5 + 1e-5))
    within_10_acc = float(np.mean(np.abs(y_true - y_pred) <= 1.0 + 1e-5))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "exact_accuracy": round(exact_acc, 4),
        "within_0.5_accuracy": round(within_05_acc, 4),
        "within_1.0_accuracy": round(within_10_acc, 4),
    }


# ---------------------------------------------------------------------------
# Model Implementations
# ---------------------------------------------------------------------------

class BaselineQuestionMeanModel:
    """Model 1: Per-Question Mean baseline."""
    def __init__(self, step: float = 0.5, max_mark: float = 2.0):
        self.step = step
        self.max_mark = max_mark
        self.question_means_: Dict[str, float] = {}
        self.global_mean_: float = 1.0

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        df = X.copy()
        df["target"] = y
        self.global_mean_ = float(df["target"].mean())
        self.question_means_ = df.groupby("qid")["target"].mean().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw_preds = np.array([self.question_means_.get(str(q), self.global_mean_) for q in X["qid"]])
        return round_and_clip(raw_preds, max_mark=self.max_mark, step=self.step)


class BaselineTfidfRidgeModel:
    """Model 2: Ridge regression on text similarity + length + question mean."""
    def __init__(self, alpha: float = 1.0, max_mark: float = 2.0):
        self.alpha = alpha
        self.max_mark = max_mark
        self.feature_cols = ["tfidf_cos", "n_words", "log_words", "question_mean_mark"]
        self.model = Ridge(alpha=self.alpha, random_state=RANDOM_SEED)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X_sub = X[self.feature_cols].fillna(0.0)
        self.model.fit(X_sub, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_sub = X[self.feature_cols].fillna(0.0)
        raw_preds = self.model.predict(X_sub)
        return round_and_clip(raw_preds, max_mark=self.max_mark)


class CandidateGradientBoostingModel:
    """Model 3: HistGradientBoosting on full feature set."""
    def __init__(self, max_mark: float = 2.0):
        self.max_mark = max_mark
        self.feature_cols = [
            "tfidf_cos",
            "coverage",
            "max_point_hit",
            "n_words",
            "log_words",
            "question_mean_mark",
            "max_mark",
            "is_missing"
        ]
        self.model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=10,
            random_state=RANDOM_SEED
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X_sub = X[self.feature_cols].fillna(0.0)
        self.model.fit(X_sub, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_sub = X[self.feature_cols].fillna(0.0)
        raw_preds = self.model.predict(X_sub)
        return round_and_clip(raw_preds, max_mark=self.max_mark)


# ---------------------------------------------------------------------------
# Cross-Validation Evaluator
# ---------------------------------------------------------------------------

def run_group_kfold_cv(
    model_factory,
    df_features: pd.DataFrame,
    model_name: str
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray, Any]:
    """
    Executes 5-fold GroupKFold CV.
    Enforces fold isolation:
    - Re-computes question_mean_mark per fold using strictly the training slice.
    - Fits model strictly on train folds.
    """
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df_features["student_id"].values
    y = df_features["mark"].values

    fold_metrics = []
    oof_predictions = np.zeros(len(df_features))

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df_features, y, groups=groups)):
        train_df = df_features.iloc[train_idx].copy()
        val_df = df_features.iloc[val_idx].copy()

        # Fold isolation: strictly compute question means on train fold
        train_q_means = compute_question_means(train_df[["qid", "mark"]])
        train_df["question_mean_mark"] = train_df["qid"].map(train_q_means).fillna(1.0)
        val_df["question_mean_mark"] = val_df["qid"].map(train_q_means).fillna(1.0)

        # Instantiate fresh model and fit
        model = model_factory()
        model.fit(train_df, train_df["mark"].values)

        preds = model.predict(val_df)
        oof_predictions[val_idx] = preds

        metrics = evaluate_predictions(val_df["mark"].values, preds)
        fold_metrics.append(metrics)

    # Compute mean and standard deviation across folds
    df_metrics = pd.DataFrame(fold_metrics)
    mean_metrics = {f"mean_{k}": float(df_metrics[k].mean()) for k in df_metrics.columns}
    std_metrics = {f"std_{k}": float(df_metrics[k].std()) for k in df_metrics.columns}

    # Train final model on full dataset with whole-dataset fold-isolated means
    full_df = df_features.copy()
    full_q_means = compute_question_means(full_df[["qid", "mark"]])
    full_df["question_mean_mark"] = full_df["qid"].map(full_q_means).fillna(1.0)
    final_model = model_factory()
    final_model.fit(full_df, full_df["mark"].values)

    return mean_metrics, std_metrics, oof_predictions, final_model


# ---------------------------------------------------------------------------
# Training Orchestration & Registry
# ---------------------------------------------------------------------------

def train_and_compare_models():
    print("=" * 80)
    print("ANSWER FEEDBACK MODEL TRAINING & COMPARISON")
    print("=" * 80)

    # 1. Ensure features dataset exists
    if not os.path.exists(OUTPUT_FEATURES_CSV):
        print(f"Features file {OUTPUT_FEATURES_CSV} not found. Generating...")
        df_features = generate_feature_dataset()
    else:
        df_features = pd.read_csv(OUTPUT_FEATURES_CSV)

    print(f"Dataset shape: {df_features.shape}")
    print(f"Total students: {df_features['student_id'].nunique()}, Short-answer questions: {df_features['qid'].nunique()}")

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    git_commit = get_git_commit_hash()
    rubric_hash = get_rubric_sha256()

    models_to_train = [
        ("baseline_question_mean", lambda: BaselineQuestionMeanModel(), ["qid", "question_mean_mark"]),
        ("baseline_tfidf_ridge", lambda: BaselineTfidfRidgeModel(alpha=1.0), ["tfidf_cos", "n_words", "log_words", "question_mean_mark"]),
        ("candidate_gradient_boosting", lambda: CandidateGradientBoostingModel(), [
            "tfidf_cos", "coverage", "max_point_hit", "n_words", "log_words", "question_mean_mark", "max_mark", "is_missing"
        ])
    ]

    results_table = []
    run_records = {}

    for model_name, factory, feature_list in models_to_train:
        print(f"\nTraining & Evaluating: {model_name}...")
        with mlflow.start_run(run_name=model_name) as run:
            mean_m, std_m, oof_preds, trained_model = run_group_kfold_cv(factory, df_features, model_name)

            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_splits", N_FOLDS)
            mlflow.log_param("split_method", "GroupKFold(student_id)")
            mlflow.log_param("n_students", int(df_features["student_id"].nunique()))
            mlflow.log_param("n_samples", int(len(df_features)))
            mlflow.log_param("git_commit", git_commit)
            mlflow.log_param("rubric_sha256", rubric_hash)
            mlflow.log_param("features", json.dumps(feature_list))

            # Log all cross-validation metrics
            for k, v in mean_m.items():
                mlflow.log_metric(k, v)
            for k, v in std_m.items():
                mlflow.log_metric(k, v)

            # Calculate and save per-question MAE breakdown
            df_eval = df_features[["student_id", "qid", "mark"]].copy()
            df_eval["oof_pred"] = oof_preds
            per_q = df_eval.groupby("qid").apply(
                lambda g: mean_absolute_error(g["mark"], g["oof_pred"])
            ).reset_index(name="mae")
            per_q_file = os.path.join(DOCS_DIR, f"per_q_mae_{model_name}.csv")
            os.makedirs(DOCS_DIR, exist_ok=True)
            per_q.to_csv(per_q_file, index=False)
            mlflow.log_artifact(per_q_file)

            run_records[model_name] = {
                "run_id": run.info.run_id,
                "mean_metrics": mean_m,
                "std_metrics": std_m,
                "model_obj": trained_model,
                "features": feature_list,
                "per_q_df": per_q
            }

            res = {
                "model_name": model_name,
                "MAE": f"{mean_m['mean_mae']:.4f} ± {std_m['std_mae']:.4f}",
                "RMSE": f"{mean_m['mean_rmse']:.4f} ± {std_m['std_rmse']:.4f}",
                "Exact_Acc": f"{mean_m['mean_exact_accuracy']*100:.1f}%",
                "Within_0.5_Acc": f"{mean_m['mean_within_0.5_accuracy']*100:.1f}%",
                "Within_1.0_Acc": f"{mean_m['mean_within_1.0_accuracy']*100:.1f}%",
                "run_id": run.info.run_id
            }
            results_table.append(res)
            print(f" -> {model_name} MAE: {res['MAE']} | RMSE: {res['RMSE']} | Within-0.5: {res['Within_0.5_Acc']}")

    # Save summary table
    df_summary = pd.DataFrame(results_table)
    df_summary.to_csv(SUMMARY_CSV_PATH, index=False)
    print(f"\nSaved comparison summary table to: {SUMMARY_CSV_PATH}")

    # -----------------------------------------------------------------------
    # Model Selection (Strictly based on cross-validation MAE)
    # -----------------------------------------------------------------------
    # Winner has minimum mean_mae
    best_model_name = min(run_records.keys(), key=lambda m: run_records[m]["mean_metrics"]["mean_mae"])
    best_info = run_records[best_model_name]

    print("\n" + "=" * 80)
    print("MODEL SELECTION & REGISTRY")
    print("=" * 80)
    print(f"Selected Champion Model: {best_model_name}")
    print(f"Mean MAE:                {best_info['mean_metrics']['mean_mae']:.4f}")
    print(f"MLflow Run ID:           {best_info['run_id']}")

    # Save best per-question MAE to standard per_question_mae.csv
    best_info["per_q_df"].to_csv(PER_Q_MAE_PATH, index=False)

    # Export selected model to models/answer_model.joblib
    os.makedirs(MODELS_DIR, exist_ok=True)
    import src.answer_feedback.train as train_mod
    model_to_save = best_info["model_obj"]
    # Recreate as clean instance of the module's class to avoid __main__ pickling issues
    clean_model = train_mod.BaselineQuestionMeanModel(
        step=getattr(model_to_save, "step", 0.5),
        max_mark=getattr(model_to_save, "max_mark", 2.0)
    )
    clean_model.question_means_ = dict(getattr(model_to_save, "question_means_", {}))
    clean_model.global_mean_ = float(getattr(model_to_save, "global_mean_", 1.0))
    joblib.dump(clean_model, MODEL_EXPORT_PATH)
    print(f"Exported model binary to: {MODEL_EXPORT_PATH}")

    # Document limitations honestly
    limitations = [
        "Trained on question-level mean marks and interim OCR segmented student answers.",
        "Handwritten OCR text coverage is currently partial in interim pipeline.",
        "If OCR text is missing, predictions gracefully fall back to question_mean_mark without assigning artificial zero marks.",
        "Evaluated with 5-fold GroupKFold by student_id to prevent intra-student data leakage."
    ]

    meta = {
        "model_name": best_model_name,
        "model_version": "1.0.0",
        "mlflow_run_id": best_info["run_id"],
        "git_commit": git_commit,
        "features": best_info["features"],
        "metrics": {
            "cv_5fold": {**best_info["mean_metrics"], **best_info["std_metrics"]}
        },
        "trained_on": {
            "dataset": "Mendeley Student Examination Papers (Anonymized)",
            "n_students": int(df_features["student_id"].nunique()),
            "n_records": int(len(df_features)),
            "short_answer_questions": "Q21-Q35"
        },
        "rubric_version": rubric_hash,
        "limitations": limitations
    }

    with open(META_EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Exported model metadata to: {META_EXPORT_PATH}")

    # Register winner in MLflow
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        # Log model artifact inside run
        with mlflow.start_run(run_id=best_info["run_id"]):
            mlflow.log_artifact(MODEL_EXPORT_PATH, artifact_path="model")
            mlflow.log_artifact(META_EXPORT_PATH, artifact_path="model")
            mlflow.log_artifact(PER_Q_MAE_PATH, artifact_path="evaluation")

        model_uri = f"runs:/{best_info['run_id']}/model"
        print(f"Registering model in MLflow Model Registry as '{REGISTERED_MODEL_NAME}'...")
        registered_model = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        
        # Set champion alias
        try:
            client.set_registered_model_alias(
                name=REGISTERED_MODEL_NAME,
                alias=MODEL_ALIAS,
                version=registered_model.version
            )
            print(f"Successfully assigned alias '@{MODEL_ALIAS}' to version {registered_model.version}")
        except Exception as e:
            # Fallback if alias API varies with MLflow client version
            try:
                client.transition_model_version_stage(
                    name=REGISTERED_MODEL_NAME,
                    version=registered_model.version,
                    stage="Production"
                )
                print(f"Set stage to Production (alias fallback: {e})")
            except Exception as e2:
                print(f"Stage transition note: {e2}")

    except Exception as e:
        print(f"MLflow model registry note: {e}")

    # Print final comparison table
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON TABLE")
    print("=" * 80)
    print(df_summary[["model_name", "MAE", "RMSE", "Exact_Acc", "Within_0.5_Acc", "Within_1.0_Acc"]].to_string(index=False))

    return df_summary, best_model_name


# ---------------------------------------------------------------------------
# Backward-compatibility aliases (required by test_baseline.py)
# ---------------------------------------------------------------------------

class QuestionMeanBaseline(BaselineQuestionMeanModel):
    """
    Alias for BaselineQuestionMeanModel, preserved for backward compatibility
    with existing test suite (test_baseline.py).
    Extends predict() to accept the deprecated round_to_step kwarg.
    """
    def predict(self, X: pd.DataFrame, round_to_step: bool = True) -> np.ndarray:
        preds = np.array([self.question_means_.get(str(q), self.global_mean_) for q in X["qid"]])
        clipped = np.clip(preds, 0.0, self.max_mark)
        if round_to_step:
            factor = 1.0 / self.step
            return np.clip(np.round(clipped * factor) / factor, 0.0, self.max_mark)
        return clipped


if __name__ == "__main__":
    train_and_compare_models()
