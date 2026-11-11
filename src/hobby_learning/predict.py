"""Prediction and inference utilities for Feature A: Hobby Learning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

LEARNING_MODEL_PATH = MODELS_DIR / "learning_model.joblib"
LEARNING_META_PATH = MODELS_DIR / "learning_meta.json"
DEMO_LEARNERS_PATH = MODELS_DIR / "demo_learners.csv"
PROBLEM_STATS_PATH = MODELS_DIR / "problem_stats.csv"
CONCEPT_STATS_PATH = MODELS_DIR / "concept_stats.csv"
HOBBY_MAP_PATH = MODELS_DIR / "hobby_map.json"

_MODEL_CACHE: Dict[str, Any] = {}
_META_CACHE: Dict[str, Any] = {}
_DEMO_LEARNERS_CACHE: Optional[pd.DataFrame] = None
_PROBLEM_STATS_CACHE: Optional[pd.DataFrame] = None
_CONCEPT_STATS_CACHE: Optional[pd.DataFrame] = None
_HOBBY_MAP_CACHE: Optional[Dict[str, Any]] = None


def load_model(model_path: Union[str, Path]) -> Any:
    """Load a serialized model from disk with in-memory caching."""
    path = Path(model_path)
    path_key = str(path.resolve())
    if path_key in _MODEL_CACHE:
        return _MODEL_CACHE[path_key]
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    _MODEL_CACHE[path_key] = model
    return model


def load_meta(meta_path: Union[str, Path] = LEARNING_META_PATH) -> Dict[str, Any]:
    """Load metadata dictionary from disk with caching."""
    path = Path(meta_path)
    path_key = str(path.resolve())
    if path_key in _META_CACHE:
        return _META_CACHE[path_key]
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    _META_CACHE[path_key] = meta
    return meta


def predict(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """Generate binary predictions from the model."""
    preds = model.predict(X)
    return np.asarray(preds, dtype=int)


def predict_proba(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """Generate probability predictions of the positive class (correct=1)."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return np.asarray(probs[:, 1], dtype=float)
        elif probs.ndim == 2 and probs.shape[1] == 1:
            return np.asarray(probs[:, 0], dtype=float)
        return np.asarray(probs, dtype=float)
    raise AttributeError("Model does not support predict_proba.")


def _get_demo_learners() -> pd.DataFrame:
    global _DEMO_LEARNERS_CACHE
    if _DEMO_LEARNERS_CACHE is None:
        if not DEMO_LEARNERS_PATH.is_file():
            raise FileNotFoundError(f"Demo learners file not found at: {DEMO_LEARNERS_PATH}")
        _DEMO_LEARNERS_CACHE = pd.read_csv(DEMO_LEARNERS_PATH)
    return _DEMO_LEARNERS_CACHE


def _get_problem_stats() -> pd.DataFrame:
    global _PROBLEM_STATS_CACHE
    if _PROBLEM_STATS_CACHE is None:
        if not PROBLEM_STATS_PATH.is_file():
            raise FileNotFoundError(f"Problem stats file not found at: {PROBLEM_STATS_PATH}")
        _PROBLEM_STATS_CACHE = pd.read_csv(PROBLEM_STATS_PATH)
    return _PROBLEM_STATS_CACHE


def _get_concept_stats() -> pd.DataFrame:
    global _CONCEPT_STATS_CACHE
    if _CONCEPT_STATS_CACHE is None:
        if not CONCEPT_STATS_PATH.is_file():
            raise FileNotFoundError(f"Concept stats file not found at: {CONCEPT_STATS_PATH}")
        _CONCEPT_STATS_CACHE = pd.read_csv(CONCEPT_STATS_PATH)
    return _CONCEPT_STATS_CACHE


def _get_hobby_map() -> Dict[str, Any]:
    global _HOBBY_MAP_CACHE
    if _HOBBY_MAP_CACHE is None:
        if not HOBBY_MAP_PATH.is_file():
            raise FileNotFoundError(f"Hobby map file not found at: {HOBBY_MAP_PATH}")
        _HOBBY_MAP_CACHE = json.loads(HOBBY_MAP_PATH.read_text(encoding="utf-8"))
    return _HOBBY_MAP_CACHE


def recommend(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate exercise correctness estimation and hobby-based learning analogy.

    Args:
        payload: Dict containing:
            - learner_id: str
            - subject: str (must be 'Mathematics')
            - topic: str
            - hobby: str
            - target_problem_id: str

    Returns:
        Envelope dict containing:
            - model_name: str
            - model_version: str
            - warnings: List[str]
            - data: Dict with p_correct, predicted_correct, weak_concepts, analogy, recommendation
    """
    subject = payload.get("subject", "").strip()
    if subject != "Mathematics":
        raise ValueError(f"Subject '{subject}' is not supported. Only 'Mathematics' is supported.")

    learner_id = payload.get("learner_id", "").strip()
    target_problem_id = payload.get("target_problem_id", "").strip()
    topic = payload.get("topic", "").strip().lower()
    hobby = payload.get("hobby", "").strip()

    warnings: List[str] = []

    # 1. Load Model & Metadata
    meta = load_meta(LEARNING_META_PATH)
    model = load_model(LEARNING_MODEL_PATH)
    global_rate = float(meta.get("global_rate", 0.675))
    attempts_median = float(meta.get("attempts_median", 1.2))

    # 2. Look up Learner History
    demo_df = _get_demo_learners()
    learner_match = demo_df[demo_df["user_id"] == learner_id]
    if not learner_match.empty:
        u_prev_n = float(learner_match.iloc[0]["u_prev_n"])
        u_prev_rate = float(learner_match.iloc[0]["u_prev_rate"])
        u_prev_attempts_mean = attempts_median
    else:
        u_prev_n = 0.0
        u_prev_rate = global_rate
        u_prev_attempts_mean = attempts_median
        warnings.append("cold start: no history for this learner")

    # 3. Look up Problem Difficulty
    prob_df = _get_problem_stats()
    prob_match = prob_df[prob_df["problem_id"] == target_problem_id]
    if not prob_match.empty:
        p_rate = float(prob_match.iloc[0]["p_rate"])
    else:
        p_rate = global_rate
        warnings.append(f"Problem '{target_problem_id}' unknown; defaulted to global success rate.")

    # 4. Look up Concept Difficulty
    conc_df = _get_concept_stats()
    conc_match = conc_df[conc_df["concept_id"] == topic]
    if not conc_match.empty:
        c_rate = float(conc_match.iloc[0]["c_rate"])
    else:
        c_rate = global_rate
        warnings.append(f"Topic/concept '{topic}' unknown; defaulted to global success rate.")

    # 5. Build Feature Vector according to meta["features"]
    feature_names = meta.get(
        "features",
        ["u_prev_n", "u_prev_rate", "u_prev_attempts_mean", "gap_hours", "n_concepts", "p_rate", "c_rate"],
    )

    feature_values: Dict[str, Any] = {
        "u_prev_n": u_prev_n,
        "u_prev_rate": u_prev_rate,
        "u_prev_attempts_mean": u_prev_attempts_mean,
        "gap_hours": 24.0,
        "n_concepts": 1.0,
        "p_rate": p_rate,
        "c_rate": c_rate,
    }

    feature_df = pd.DataFrame([{col: feature_values.get(col, 0.0) for col in feature_names}])

    # 6. Predict Probability and Binary Class
    classification_threshold = float(meta.get("classification_threshold", 0.5))
    probs = predict_proba(model, feature_df)
    p_correct = float(np.clip(probs[0], 0.0, 1.0))
    predicted_correct = bool(p_correct >= classification_threshold)

    # 7. Identify Weak Concepts
    weak_concepts = [topic] if c_rate < 0.65 else []

    # 8. Retrieve Hobby Analogy from hobby_map.json
    hobby_map = _get_hobby_map()
    hobby_entries = hobby_map.get(hobby, {})
    analogy_obj = hobby_entries.get(topic)

    analogy_result: Optional[Dict[str, Any]] = None
    if isinstance(analogy_obj, dict):
        analogy_result = analogy_obj
        if not analogy_obj.get("valid", True):
            reason = analogy_obj.get("reason", "No valid mathematically accurate analogy found.")
            warnings.append(f"Hobby analogy for '{hobby}' on topic '{topic}' is marked invalid: {reason}")
    else:
        warnings.append(f"No analogy mapping available for hobby '{hobby}' and topic '{topic}'.")

    # 9. Recommendation Text
    if predicted_correct:
        recommendation_msg = (
            f"Learner is predicted to solve {target_problem_id} with high confidence ({p_correct:.1%}). "
            f"Ready for advanced exercise."
        )
    else:
        recommendation_msg = (
            f"Learner may find {target_problem_id} challenging ({p_correct:.1%}). "
            f"Reviewing {topic} with relatable analogies is recommended before attempting."
        )

    return {
        "model_name": meta.get("model_name", "study-assistant-learning-model"),
        "model_version": str(meta.get("model_version", "1.0.0")),
        "warnings": warnings,
        "data": {
            "p_correct": round(p_correct, 4),
            "predicted_correct": predicted_correct,
            "weak_concepts": weak_concepts,
            "analogy": analogy_result,
            "recommendation": recommendation_msg,
        },
    }
