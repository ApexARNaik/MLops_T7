"""
Production Prediction and Scoring Module for Answer Feedback
Owner: Member 2 (AV)
Project: AI-Powered Student Study Assistant MLOps Project
Path: src/answer_feedback/predict.py

Exposes the production evaluation interface:
    evaluate(answers: dict[str, str], ocr_conf: dict[str, float] | None = None) -> dict

Features & Requirements:
- Validates input question IDs against rubric.json (raises ValueError for unknown QIDs).
- Never leaks or includes raw student answer text in logs.
- Uses models/answer_model.joblib, models/answer_meta.json, and data/answer_feedback/processed/rubric.json.
- Deterministic, template-based feedback (no LLM, no external APIs).
- Clips predictions to [0, max_mark] and rounds to 0.5 increments.
- Sets approved_mark = None, status = 'pending_teacher_review'.
- Sets human_review_required = True when:
    * OCR confidence < 0.6
    * Word count < 3
    * Answer is empty or blank
- Returns the exact prediction envelope expected by FastAPI.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

from src.answer_feedback.feedback import (
    compute_point_breakdown,
    build_deterministic_feedback,
)
from src.answer_feedback.features import (
    compute_rubric_coverage,
    normalize_text,
    compute_word_counts,
    TfidfSimilarityEngine,
)

# Logger - configured without logging raw student text
logger = logging.getLogger(__name__)

# Base paths relative to repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
MODEL_PATH = MODELS_DIR / "answer_model.joblib"
META_PATH = MODELS_DIR / "answer_meta.json"
RUBRIC_PATH = REPO_ROOT / "data" / "answer_feedback" / "processed" / "rubric.json"

ALLOWED_MARKS = [0.0, 0.5, 1.0, 1.5, 2.0]
OCR_CONF_REVIEW_THRESHOLD = 0.60
MIN_WORD_COUNT = 3

# Cached in-memory artifacts
_RUBRIC_CACHE: Optional[Dict[str, Any]] = None
_META_CACHE: Optional[Dict[str, Any]] = None
_MODEL_CACHE: Optional[Any] = None
_TFIDF_ENGINE_CACHE: Optional[TfidfSimilarityEngine] = None


def load_rubric() -> Dict[str, Any]:
    """Loads and caches rubric.json."""
    global _RUBRIC_CACHE
    if _RUBRIC_CACHE is None:
        if not RUBRIC_PATH.exists():
            raise FileNotFoundError(f"Rubric file not found at: {RUBRIC_PATH}")
        with open(RUBRIC_PATH, "r", encoding="utf-8") as f:
            _RUBRIC_CACHE = json.load(f)
    return _RUBRIC_CACHE


def load_metadata() -> Dict[str, Any]:
    """Loads and caches answer_meta.json."""
    global _META_CACHE
    if _META_CACHE is None:
        if not META_PATH.exists():
            raise FileNotFoundError(f"Model metadata not found at: {META_PATH}")
        with open(META_PATH, "r", encoding="utf-8") as f:
            _META_CACHE = json.load(f)
    return _META_CACHE


def load_model() -> Any:
    """Loads and caches the trained Joblib model."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")
        # Ensure class definitions from training module are importable for pickle
        try:
            from src.answer_feedback.train import (  # noqa: F401
                BaselineQuestionMeanModel,
                QuestionMeanBaseline,
                BaselineTfidfRidgeModel,
                CandidateGradientBoostingModel,
            )
        except ImportError:
            pass
        _MODEL_CACHE = joblib.load(MODEL_PATH)
        logger.info("Loaded answer scoring model artifact.")
    return _MODEL_CACHE


def load_tfidf_engine() -> TfidfSimilarityEngine:
    """Initializes and caches TF-IDF similarity engine on rubric references."""
    global _TFIDF_ENGINE_CACHE
    if _TFIDF_ENGINE_CACHE is None:
        rubric = load_rubric()
        _TFIDF_ENGINE_CACHE = TfidfSimilarityEngine(rubric=rubric)
    return _TFIDF_ENGINE_CACHE


def snap_to_allowed(mark: float, max_mark: float = 2.0) -> float:
    """
    Rounds predicted mark to allowed step (0.5 increments)
    and bounds it within [0.0, max_mark].
    """
    clipped = max(0.0, min(float(mark), float(max_mark)))
    return float(min(ALLOWED_MARKS, key=lambda v: abs(v - clipped)))


def evaluate(
    answers: dict[str, str],
    ocr_conf: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Production evaluation function integrated with FastAPI endpoint.

    Parameters:
      answers: mapping from question ID (e.g. 'Q21') to student answer text
      ocr_conf: optional mapping from question ID to OCR confidence float [0.0, 1.0]

    Returns:
      Envelope matching API specification:
      {
          "model_name": "study-assistant-answer-scoring-model",
          "model_version": "...",
          "warnings": [],
          "data": {
              "results": [
                  {
                      "qid": "Q21",
                      "estimated_mark": 2.0,
                      "max_mark": 2.0,
                      "approved_mark": None,
                      "status": "pending_teacher_review",
                      "human_review_required": False,
                      "ocr_confidence": 0.85,
                      "matched_points": [...],
                      "missing_points": [...],
                      "deduction_rationale": "...",
                      "improvement_suggestion": "...",
                      "evidence_refs": [...],
                      "feedback_text": "..."
                  },
                  ...
              ]
          }
      }

    Raises:
      ValueError: if an unknown question ID is provided (mapped to 422 in API).
    """
    rubric = load_rubric()
    metadata = load_metadata()
    model = load_model()
    tfidf_engine = load_tfidf_engine()

    # Log request receipt safely without logging raw student text
    qids = list(answers.keys())
    logger.info("Evaluating answer sheet submission with %d question(s).", len(qids))

    # 1. Validate all input question IDs against rubric.json
    for qid in qids:
        if qid not in rubric:
            # Raise ValueError with qid name only (never student text)
            raise ValueError(f"Unknown question ID '{qid}'. Must be one of: {sorted(list(rubric.keys()))}")

    warnings: List[str] = []
    results: List[Dict[str, Any]] = []

    for qid, raw_answer in answers.items():
        rubric_item = rubric[qid]
        max_mark = float(rubric_item.get("max_mark", 2.0))
        conf_val: Optional[float] = None
        if ocr_conf is not None and qid in ocr_conf:
            conf_val = float(ocr_conf[qid])

        # Analyze answer text characteristics
        norm_answer = normalize_text(raw_answer)
        words = norm_answer.split() if norm_answer else []
        n_words = len(words)
        is_blank = n_words == 0

        # Review trigger conditions
        review_reasons: List[str] = []
        if is_blank:
            review_reasons.append("Answer is empty or unreadable")
        elif n_words < MIN_WORD_COUNT:
            review_reasons.append(f"Answer contains only {n_words} word(s) (minimum {MIN_WORD_COUNT} required)")

        if conf_val is not None and conf_val < OCR_CONF_REVIEW_THRESHOLD:
            review_reasons.append(f"Low OCR confidence ({conf_val:.2f} < {OCR_CONF_REVIEW_THRESHOLD})")

        human_review_required = len(review_reasons) > 0

        # Calculate matched and missing rubric points
        matched_points, missing_points = compute_point_breakdown(raw_answer, rubric_item)

        # Mark scoring logic:
        # Never silently assign high/confident score to empty answer.
        if is_blank:
            # Empty / blank answer: assigned 0.0 with review required
            estimated_mark = 0.0
        else:
            # Feature calculation: match training pipeline
            coverage, max_point_hit = compute_rubric_coverage(raw_answer, rubric_item)
            tfidf_cos = tfidf_engine.compute_similarity(qid, raw_answer)

            # Predict using model: pass dataframe matching feature columns
            feat_df = pd.DataFrame([{
                "qid": qid,
                "tfidf_cos": tfidf_cos,
                "coverage": coverage,
                "max_point_hit": max_point_hit,
                "n_words": n_words,
                "log_words": round(float(np.log1p(n_words)), 4),
                "question_mean_mark": 1.0,
                "max_mark": max_mark,
                "is_missing": 0
            }])

            try:
                raw_pred = float(model.predict(feat_df)[0])
                # If rubric matched 0 points and text is short, penalize raw model prediction
                if max_point_hit == 0 and coverage == 0.0:
                    raw_pred = min(raw_pred, 0.5)
            except Exception:
                # Fallback to rubric coverage score
                raw_pred = coverage * max_mark

            estimated_mark = snap_to_allowed(raw_pred, max_mark=max_mark)

        # Deterministic feedback generation
        feedback_details = build_deterministic_feedback(
            qid=qid,
            answer_text=raw_answer,
            estimated_mark=estimated_mark,
            max_mark=max_mark,
            matched_points=matched_points,
            missing_points=missing_points,
            rubric_item=rubric_item,
            requires_review=human_review_required,
            review_reasons=review_reasons,
        )

        # Evidence references pointing to question and answerkey.txt
        evidence_refs = [
            f"answerkey.txt#Q{qid.lstrip('Q')}",
            f"rubric.json#{qid}",
        ]

        result_entry: Dict[str, Any] = {
            "qid": qid,
            "estimated_mark": estimated_mark,
            "max_mark": max_mark,
            "approved_mark": None,
            "status": "pending_teacher_review",
            "human_review_required": human_review_required,
            "ocr_confidence": conf_val,
            "matched_points": matched_points,
            "missing_points": missing_points,
            "deduction_rationale": feedback_details["deduction_rationale"],
            "improvement_suggestion": feedback_details["improvement_suggestion"],
            "evidence_refs": evidence_refs,
            "feedback_text": feedback_details["feedback_text"],
        }
        results.append(result_entry)

    # Registered model naming convention
    model_name = metadata.get("registered_model_name", "study-assistant-answer-scoring-model")
    model_version = str(metadata.get("model_version", "1.0.0"))

    response = {
        "model_name": model_name,
        "model_version": model_version,
        "warnings": warnings,
        "data": {
            "results": results
        }
    }
    return response


def rubric_score(answer_text: Optional[str], rubric_item: Dict[str, Any]) -> float:
    """Computes a rubric-based score for test compatibility."""
    if not answer_text or not str(answer_text).strip():
        return np.nan
    coverage, _ = compute_rubric_coverage(answer_text, rubric_item)
    return snap_to_allowed(coverage * float(rubric_item.get("max_mark", 2.0)))


# Backwards compatibility class for tests/answer_feedback/
class AnswerScorer:
    """Wrapper class retained for backward compatibility with previous test suites."""
    def __init__(self, *args, **kwargs):
        self.rubric = load_rubric()

    def predict(self, qid: str, answer_text: Optional[str], ocr_conf: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        if qid not in self.rubric:
            return {
                "qid": qid,
                "predicted_mark": np.nan,
                "prediction_method": "unknown_qid",
                "requires_review": True,
                "error": f"Unknown question {qid}"
            }
        
        if not answer_text or not str(answer_text).strip():
            return {
                "qid": qid,
                "predicted_mark": np.nan,
                "prediction_method": "unattempted",
                "rubric_score": np.nan,
                "coverage": 0.0,
                "max_point_hit": 0,
                "requires_review": True,
                "ocr_conf": ocr_conf,
            }

        conf_map = {qid: ocr_conf} if ocr_conf is not None else None
        res = evaluate({qid: answer_text}, ocr_conf=conf_map)
        item = res["data"]["results"][0]
        return {
            "qid": qid,
            "predicted_mark": item["estimated_mark"],
            "prediction_method": "model",
            "rubric_score": item["estimated_mark"],
            "coverage": len(item["matched_points"]) / max(1, len(item["matched_points"]) + len(item["missing_points"])),
            "max_point_hit": 1 if item["matched_points"] else 0,
            "requires_review": item["human_review_required"],
            "ocr_conf": ocr_conf,
        }
