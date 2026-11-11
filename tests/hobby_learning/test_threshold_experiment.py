"""Unit tests for threshold selection, validation splitting, and threshold classification."""

import numpy as np
import pandas as pd
import pytest

from src.hobby_learning.threshold_experiment import (
    chronological_three_way_split,
    classify_probabilities,
    compute_metrics_at_threshold,
    select_best_threshold,
)
from src.hobby_learning.predict import recommend, load_meta, LEARNING_META_PATH


@pytest.fixture
def mock_chronological_df() -> pd.DataFrame:
    """Fixture providing ordered interaction data."""
    n = 100
    return pd.DataFrame(
        {
            "order_id": list(range(100, 100 + n)),
            "user_id": [f"u{i % 5}" for i in range(n)],
            "problem_id": [f"p{i % 10}" for i in range(n)],
            "correct": [1 if i % 3 != 0 else 0 for i in range(n)],
            "user_prev_count": list(range(n)),
            "user_prev_accuracy": [0.7] * n,
            "user_prev_attempts_mean": [1.2] * n,
            "user_prev_hints_mean": [0.5] * n,
            "user_skill_prev_count": list(range(n)),
            "user_skill_prev_accuracy": [0.7] * n,
            "n_skill_components": [1.0] * n,
        }
    )


def test_chronological_three_way_split_boundaries(mock_chronological_df: pd.DataFrame):
    """Test 70/15/15 chronological split preserves order with zero leakage."""
    shuffled = mock_chronological_df.sample(frac=1.0, random_state=42)
    train_df, val_df, test_df = chronological_three_way_split(
        shuffled, train_size=0.70, val_size=0.15, test_size=0.15, order_col="order_id"
    )

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Strictly monotonically increasing order_ids across splits
    assert train_df["order_id"].max() < val_df["order_id"].min()
    assert val_df["order_id"].max() < test_df["order_id"].min()


def test_classify_probabilities_threshold_boundary():
    """Verify probabilities equal to threshold are classified as positive, below as negative."""
    probs = np.array([0.39, 0.40, 0.41, 0.499, 0.50, 0.501])

    # Threshold 0.40
    preds_040 = classify_probabilities(probs, threshold=0.40)
    assert np.array_equal(preds_040, np.array([0, 1, 1, 1, 1, 1]))

    # Threshold 0.50
    preds_050 = classify_probabilities(probs, threshold=0.50)
    assert np.array_equal(preds_050, np.array([0, 0, 0, 0, 1, 1]))


def test_compute_metrics_at_threshold():
    """Verify metrics computation matches expected definitions."""
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.2, 0.4, 0.6, 0.8])

    # At threshold 0.50: preds = [0, 0, 1, 1] -> perfect
    metrics_050 = compute_metrics_at_threshold(y_true, probs, threshold=0.50)
    assert metrics_050["accuracy"] == 1.0
    assert metrics_050["precision"] == 1.0
    assert metrics_050["recall"] == 1.0
    assert metrics_050["f1_score"] == 1.0
    assert metrics_050["specificity"] == 1.0
    assert metrics_050["false_positive_count"] == 0
    assert metrics_050["false_negative_count"] == 0

    # At threshold 0.30: preds = [0, 1, 1, 1] -> 1 false positive
    metrics_030 = compute_metrics_at_threshold(y_true, probs, threshold=0.30)
    assert metrics_030["false_positive_count"] == 1
    assert metrics_030["false_negative_count"] == 0
    assert metrics_030["precision"] == 2 / 3
    assert metrics_030["recall"] == 1.0


def test_select_best_threshold_uses_validation_metrics_only():
    """Verify selection uses only provided sweep results and respects F1 + precision tiebreaker."""
    mock_sweep = [
        {"threshold": 0.40, "f1_score": 0.80, "precision": 0.70, "recall": 0.93, "balanced_accuracy": 0.65},
        {"threshold": 0.50, "f1_score": 0.80, "precision": 0.75, "recall": 0.86, "balanced_accuracy": 0.68},
        {"threshold": 0.60, "f1_score": 0.78, "precision": 0.80, "recall": 0.76, "balanced_accuracy": 0.70},
    ]

    selected = select_best_threshold(mock_sweep)
    # Both 0.40 and 0.50 have F1=0.80, tie-breaker prefers higher precision (0.50 with prec=0.75)
    assert selected["selected_best_f1"]["threshold"] == 0.50
    assert selected["highest_precision"]["threshold"] == 0.60
    assert selected["highest_recall"]["threshold"] == 0.40
    assert selected["highest_balanced_accuracy"]["threshold"] == 0.60


def test_serving_model_metadata_and_threshold_behavior():
    """Verify that serving metadata contains configurable threshold and predict.recommend respects it."""
    meta = load_meta(LEARNING_META_PATH)
    assert "classification_threshold" in meta
    assert meta["classification_threshold"] == 0.5

    payload = {
        "learner_id": "U_test",
        "subject": "Mathematics",
        "topic": "fractions",
        "hobby": "Gaming",
        "target_problem_id": "P_test",
    }
    rec = recommend(payload)
    p_correct = rec["data"]["p_correct"]
    predicted_correct = rec["data"]["predicted_correct"]

    # Verify decision matches threshold
    assert predicted_correct == (p_correct >= meta["classification_threshold"])
