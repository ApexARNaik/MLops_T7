"""Prediction and inference utilities for Feature A: Hobby Learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union
import joblib
import numpy as np
import pandas as pd


def load_model(model_path: Union[str, Path]) -> Any:
    """Load a serialized model from disk.

    Args:
        model_path: Path to the serialized model file (.joblib).

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """Generate binary predictions from the model.

    Args:
        model: Fitted estimator.
        X: Feature matrix.

    Returns:
        1D numpy array of binary predictions (0 or 1).
    """
    preds = model.predict(X)
    return np.asarray(preds, dtype=int)


def predict_proba(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """Generate probability predictions of the positive class (correct=1).

    Args:
        model: Fitted estimator.
        X: Feature matrix.

    Returns:
        1D numpy array of probabilities.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return np.asarray(probs[:, 1], dtype=float)
        elif probs.ndim == 2 and probs.shape[1] == 1:
            return np.asarray(probs[:, 0], dtype=float)
        return np.asarray(probs, dtype=float)
    raise AttributeError("Model does not support predict_proba.")

