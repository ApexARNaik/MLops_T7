"""Centralized model artifact paths and loader utilities.

Separates:
1. Production Serving Models:
   - Feature A: models/learning_model.joblib (recommendation model)
   - Feature B: models/answer_model.joblib (answer feedback model)
2. Feature A ASSISTments Experimentation/Candidate Models:
   - models/hobby_learning/baseline_model.joblib
   - models/hobby_learning/logistic_regression.joblib
   - models/hobby_learning/hist_gradient_boosting.joblib
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
HOBBY_LEARNING_MODELS_DIR = MODELS_DIR / "hobby_learning"

# Production serving model artifacts and metadata
SERVING_ARTIFACTS: Dict[str, Path] = {
    "learning": MODELS_DIR / os.getenv("HOBBY_LEARNING_MODEL_FILENAME", "learning_model.joblib"),
    "answer": MODELS_DIR / os.getenv("ANSWER_FEEDBACK_MODEL_FILENAME", "answer_model.joblib"),
}

SERVING_METADATA: Dict[str, Path] = {
    "learning": MODELS_DIR / os.getenv("HOBBY_LEARNING_META_FILENAME", "learning_meta.json"),
    "answer": MODELS_DIR / os.getenv("ANSWER_FEEDBACK_META_FILENAME", "answer_meta.json"),
}

# Supporting production files
SUPPORTING_FILES: Dict[str, Path] = {
    "hobby_map": MODELS_DIR / "hobby_map.json",
    "problem_stats": MODELS_DIR / "problem_stats.csv",
    "concept_stats": MODELS_DIR / "concept_stats.csv",
    "demo_learners": MODELS_DIR / "demo_learners.csv",
    "rubric": REPO_ROOT / "data" / "answer_feedback" / "processed" / "rubric.json",
}

# In-memory artifact cache
_LOADED_MODELS: Dict[str, Any] = {}
_LOADED_METADATA: Dict[str, Dict[str, Any]] = {}


def get_model_path(model_key: str) -> Path:
    """Return the absolute Path for a given production model key."""
    if model_key not in SERVING_ARTIFACTS:
        raise KeyError(f"Unknown model key '{model_key}'. Valid keys: {list(SERVING_ARTIFACTS.keys())}")
    return SERVING_ARTIFACTS[model_key]


def get_metadata_path(model_key: str) -> Path:
    """Return the absolute Path for a given production model's metadata."""
    if model_key not in SERVING_METADATA:
        raise KeyError(f"Unknown model key '{model_key}'. Valid keys: {list(SERVING_METADATA.keys())}")
    return SERVING_METADATA[model_key]


def load_serving_model(model_key: str, force_reload: bool = False) -> Any:
    """Load and cache a production serving model artifact.

    Args:
        model_key: 'learning' or 'answer'
        force_reload: Whether to bypass cache

    Returns:
        Loaded model estimator

    Raises:
        FileNotFoundError: If model file does not exist.
        RuntimeError: If artifact is corrupt or unpicklable.
    """
    path = get_model_path(model_key)
    path_str = str(path.resolve())

    if not force_reload and path_str in _LOADED_MODELS:
        return _LOADED_MODELS[path_str]

    if not path.is_file():
        raise FileNotFoundError(f"Serving model artifact for '{model_key}' not found at: {path}")

    try:
        model = joblib.load(path)
        _LOADED_MODELS[path_str] = model
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load serving model artifact at {path}: {exc}") from exc


def load_serving_metadata(model_key: str, force_reload: bool = False) -> Dict[str, Any]:
    """Load and validate production serving metadata.

    Args:
        model_key: 'learning' or 'answer'
        force_reload: Whether to bypass cache

    Returns:
        Validated metadata dictionary

    Raises:
        FileNotFoundError: If metadata file does not exist.
        ValueError: If metadata structure is invalid.
    """
    path = get_metadata_path(model_key)
    path_str = str(path.resolve())

    if not force_reload and path_str in _LOADED_METADATA:
        return _LOADED_METADATA[path_str]

    if not path.is_file():
        raise FileNotFoundError(f"Metadata file for '{model_key}' not found at: {path}")

    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Metadata file at {path} is not valid JSON: {exc}") from exc

    if not isinstance(meta, dict):
        raise ValueError(f"Metadata at {path} must be a JSON object.")

    model_name = meta.get("model_name")
    model_version = meta.get("model_version")

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(f"Metadata at {path} missing valid 'model_name'.")

    if model_version is None or isinstance(model_version, bool) or not str(model_version).strip():
        raise ValueError(f"Metadata at {path} missing valid 'model_version'.")

    meta["model_version"] = str(model_version)
    _LOADED_METADATA[path_str] = meta
    return meta


def check_system_readiness() -> Tuple[bool, Dict[str, Any]]:
    """Validate readiness of all production serving models and supporting files.

    Returns:
        Tuple of (is_ready: bool, details_dict: Dict[str, Any])
    """
    missing_files: list[str] = []
    errors: list[str] = []
    models_info: list[dict[str, str]] = []

    # 1. Verify serving models and metadata
    for key in SERVING_ARTIFACTS:
        model_p = get_model_path(key)
        meta_p = get_metadata_path(key)

        if not model_p.is_file():
            missing_files.append(model_p.name)
        if not meta_p.is_file():
            missing_files.append(meta_p.name)

        if model_p.is_file() and meta_p.is_file():
            try:
                meta = load_serving_metadata(key)
                _ = load_serving_model(key)
                models_info.append(
                    {
                        "model_name": meta["model_name"],
                        "model_version": meta["model_version"],
                    }
                )
            except Exception as e:
                errors.append(f"{key}: {e}")

    # 2. Verify supporting files
    for name, p in SUPPORTING_FILES.items():
        if not p.is_file():
            missing_files.append(p.name)

    is_ready = len(missing_files) == 0 and len(errors) == 0
    details = {
        "is_ready": is_ready,
        "missing": missing_files,
        "errors": errors,
        "models": models_info,
    }
    return is_ready, details
