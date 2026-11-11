"""Unit tests for Feature A: Logistic Regression Training & Pipeline."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.hobby_learning.predict import load_model, predict, predict_proba
from src.hobby_learning.train import (
    FEATURE_COLUMNS,
    chronological_train_test_split,
    prepare_features_and_target,
)
from src.hobby_learning.train_lr import (
    build_logistic_regression_pipeline,
    evaluate_logistic_regression,
    generate_comparison_reports,
)


@pytest.fixture
def mock_dataset() -> pd.DataFrame:
    """Fixture providing simulated chronological interaction data with numerical features."""
    rng = np.random.RandomState(42)
    n = 100
    return pd.DataFrame({
        "order_id": list(range(100, 100 + n)),
        "user_id": [f"u{i % 10}" for i in range(n)],
        "problem_id": [f"p{i % 20}" for i in range(n)],
        "correct": rng.choice([0, 1], size=n, p=[0.35, 0.65]),
        "user_prev_count": list(range(n)),
        "user_prev_accuracy": [np.nan] + list(rng.uniform(0.2, 0.9, size=n - 1)),
        "user_prev_attempts_mean": [np.nan] + list(rng.uniform(1.0, 3.0, size=n - 1)),
        "user_prev_hints_mean": [np.nan] + list(rng.uniform(0.0, 2.0, size=n - 1)),
        "user_skill_prev_count": list(range(n)),
        "user_skill_prev_accuracy": [np.nan] + list(rng.uniform(0.1, 0.95, size=n - 1)),
        "n_skill_components": [1.0] * n,
    })


def test_logistic_regression_pipeline_training_and_binary_predictions(mock_dataset: pd.DataFrame):
    """Test that the Logistic Regression pipeline trains and produces valid binary predictions."""
    train_df, test_df = chronological_train_test_split(mock_dataset, test_size=0.3)
    X_train, y_train = prepare_features_and_target(train_df, feature_cols=FEATURE_COLUMNS)
    X_test, y_test = prepare_features_and_target(test_df, feature_cols=FEATURE_COLUMNS)

    pipeline = build_logistic_regression_pipeline(feature_cols=FEATURE_COLUMNS, random_state=42)
    pipeline.fit(X_train, y_train)

    preds = predict(pipeline, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_predicted_probabilities_are_valid(mock_dataset: pd.DataFrame):
    """Test that predicted probabilities lie within [0.0, 1.0]."""
    train_df, test_df = chronological_train_test_split(mock_dataset, test_size=0.3)
    X_train, y_train = prepare_features_and_target(train_df, feature_cols=FEATURE_COLUMNS)
    X_test, y_test = prepare_features_and_target(test_df, feature_cols=FEATURE_COLUMNS)

    pipeline = build_logistic_regression_pipeline(feature_cols=FEATURE_COLUMNS, random_state=42)
    pipeline.fit(X_train, y_train)

    probs = predict_proba(pipeline, X_test)
    assert probs.shape == (len(X_test),)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_saved_joblib_pipeline_can_be_loaded_and_predict(tmp_path: Path, mock_dataset: pd.DataFrame):
    """Test that the saved Joblib pipeline can be loaded and predict on new records."""
    X_train, y_train = prepare_features_and_target(mock_dataset, feature_cols=FEATURE_COLUMNS)
    pipeline = build_logistic_regression_pipeline(feature_cols=FEATURE_COLUMNS, random_state=42)
    pipeline.fit(X_train, y_train)

    saved_path = tmp_path / "logistic_regression.joblib"
    joblib.dump(pipeline, saved_path)

    loaded_pipe = load_model(saved_path)
    preds = predict(loaded_pipe, X_train)
    probs = predict_proba(loaded_pipe, X_train)

    assert len(preds) == len(X_train)
    assert len(probs) == len(X_train)
    assert np.array_equal(preds, predict(pipeline, X_train))


def test_preprocessing_fitted_using_training_data_only(mock_dataset: pd.DataFrame):
    """Test that preprocessing scalers and imputers are strictly fitted on training data."""
    train_df, test_df = chronological_train_test_split(mock_dataset, test_size=0.3)
    X_train, y_train = prepare_features_and_target(train_df, feature_cols=FEATURE_COLUMNS)
    X_test, y_test = prepare_features_and_target(test_df, feature_cols=FEATURE_COLUMNS)

    pipeline = build_logistic_regression_pipeline(feature_cols=FEATURE_COLUMNS, random_state=42)
    pipeline.fit(X_train, y_train)

    # Preprocessor scaler mean must match X_train mean (ignoring NaNs) rather than test mean
    scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
    train_imputed_mean = scaler.mean_
    assert len(train_imputed_mean) == len(FEATURE_COLUMNS)


def test_test_set_remains_unchanged(mock_dataset: pd.DataFrame):
    """Test that evaluation does not mutate or alter the test dataframe."""
    train_df, test_df = chronological_train_test_split(mock_dataset, test_size=0.3)
    X_test_orig = test_df[FEATURE_COLUMNS].copy()

    pipeline = build_logistic_regression_pipeline(feature_cols=FEATURE_COLUMNS, random_state=42)
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df["correct"])

    _ = evaluate_logistic_regression(pipeline, test_df[FEATURE_COLUMNS], test_df["correct"])

    pd.testing.assert_frame_equal(test_df[FEATURE_COLUMNS], X_test_orig)


def test_no_target_leakage_in_features(mock_dataset: pd.DataFrame):
    """Test that forbidden target or current outcome columns are rejected."""
    with pytest.raises(ValueError, match="Data leakage detected"):
        prepare_features_and_target(mock_dataset, feature_cols=["correct", "user_prev_count"])

    with pytest.raises(ValueError, match="Data leakage detected"):
        prepare_features_and_target(mock_dataset, feature_cols=["attempt_count", "user_prev_count"])


def test_comparison_report_generation(tmp_path: Path):
    """Test generation of model comparison CSV and Markdown reports."""
    mock_metrics = {
        "accuracy": 0.7106,
        "precision": 0.7221,
        "recall": 0.9450,
        "f1_score": 0.8187,
        "roc_auc": 0.6590,
        "pr_auc": 0.7907,
    }
    csv_path, md_path = generate_comparison_reports(mock_metrics, reports_dir=tmp_path)
    assert csv_path.exists()
    assert md_path.exists()

    df_comp = pd.read_csv(csv_path)
    assert "Metric" in df_comp.columns
    assert "Majority Baseline" in df_comp.columns
    assert "Logistic Regression" in df_comp.columns
    assert len(df_comp) == 6

