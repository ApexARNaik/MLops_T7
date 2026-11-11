"""Unit tests for Feature A: Hobby Learning Baseline Training Pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.hobby_learning.predict import load_model, predict, predict_proba
from src.hobby_learning.train import (
    chronological_train_test_split,
    evaluate_predictions,
    prepare_features_and_target,
    train_majority_baseline,
)


@pytest.fixture
def mock_interactions_df() -> pd.DataFrame:
    """Fixture providing chronological interaction data with deliberate ordering."""
    return pd.DataFrame({
        "order_id": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "user_id": ["u1"] * 10,
        "problem_id": [f"p{i}" for i in range(10)],
        "correct": [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
        "user_prev_count": list(range(10)),
        "user_prev_accuracy": [np.nan] + [0.8] * 9,
        "user_prev_attempts_mean": [np.nan] + [1.2] * 9,
        "user_prev_hints_mean": [np.nan] + [0.5] * 9,
        "user_skill_prev_count": list(range(10)),
        "user_skill_prev_accuracy": [np.nan] + [0.8] * 9,
        "n_skill_components": [1] * 10,
    })


def test_chronological_split_no_future_leakage(mock_interactions_df: pd.DataFrame):
    """Test that train/test split strictly follows chronological order."""
    # Deliberately shuffle before splitting
    shuffled = mock_interactions_df.sample(frac=1.0, random_state=42)
    train_df, test_df = chronological_train_test_split(shuffled, test_size=0.3, order_col="order_id")

    assert len(train_df) == 7
    assert len(test_df) == 3

    # All train order_ids must be strictly less than all test order_ids
    assert train_df["order_id"].max() < test_df["order_id"].min()
    assert list(test_df["order_id"]) == [80, 90, 100]


def test_feature_target_preparation_prevents_leakage(mock_interactions_df: pd.DataFrame):
    """Test that target or outcome metrics are rejected as input features."""
    X, y = prepare_features_and_target(mock_interactions_df)
    assert "correct" not in X.columns
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(mock_interactions_df)

    # Attempting to leak current outcome must raise ValueError
    with pytest.raises(ValueError, match="Data leakage detected"):
        prepare_features_and_target(mock_interactions_df, feature_cols=["correct", "user_prev_count"])

    with pytest.raises(ValueError, match="Data leakage detected"):
        prepare_features_and_target(mock_interactions_df, feature_cols=["attempt_count", "user_prev_count"])


def test_training_executes_successfully_and_predicts(mock_interactions_df: pd.DataFrame):
    """Test that DummyClassifier trains and generates predictions with correct shape & binary values."""
    train_df, test_df = chronological_train_test_split(mock_interactions_df, test_size=0.4)
    X_train, y_train = prepare_features_and_target(train_df)
    X_test, y_test = prepare_features_and_target(test_df)

    clf = train_majority_baseline(X_train, y_train)

    preds = predict(clf, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X_test),)
    # Output must be binary classes (0 or 1)
    assert set(np.unique(preds)).issubset({0, 1})

    probs = predict_proba(clf, X_test)
    assert probs.shape == (len(X_test),)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_evaluation_metrics_generation():
    """Test that metrics, confusion matrix, and classification report are correctly computed."""
    y_true = np.array([1, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 1, 1])  # Majority baseline prediction
    y_prob = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    metrics, cm, report_dict, report_text = evaluate_predictions(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics

    # Accuracy = 3 / 5 = 0.6
    assert pytest.approx(metrics["accuracy"]) == 0.6
    assert pytest.approx(metrics["recall"]) == 1.0
    assert cm.shape == (2, 2)
    assert "precision" in report_text


def test_model_loading_and_prediction(tmp_path: Path, mock_interactions_df: pd.DataFrame):
    """Test saving and loading interface using predict.py."""
    X_train, y_train = prepare_features_and_target(mock_interactions_df)
    clf = train_majority_baseline(X_train, y_train)

    import joblib
    saved_path = tmp_path / "test_model.joblib"
    joblib.dump(clf, saved_path)

    loaded_clf = load_model(saved_path)
    preds = predict(loaded_clf, X_train)
    assert len(preds) == len(X_train)
    assert (preds == y_train.mode()[0]).all()

