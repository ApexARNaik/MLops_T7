"""
Prediction Module for Answer Feedback
Owner: Member 2 (AV)
Project: AI-Powered Student Study Assistant MLOps Project
Path: src/answer_feedback/predict.py

Given a student's answer_text and qid, computes a predicted mark using:
  1. Rubric keyword matching score (rule-based, always available)
  2. Loaded champion model (BaselineQuestionMeanModel or ML model)

Outputs a predicted mark in the allowed set {0, 0.5, 1.0, 1.5, 2.0} and
a confidence indicator derived from OCR confidence (if available).

Constraints:
- Never uses Corrected_Pdf.
- Never uses the student's true mark as a feature.
- Missing/low-confidence answers are flagged for review, never zero-filled.
- All feature calculations go through the same features.py functions used in training.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from src.answer_feedback.features import (
    TfidfSimilarityEngine,
    extract_features_for_pair,
    compute_rubric_coverage,
)

# Default paths
DEFAULT_RUBRIC_JSON = os.path.join("data", "answer_feedback", "processed", "rubric.json")
DEFAULT_MODEL_PATH = os.path.join("models", "answer_model.joblib")
DEFAULT_MARKS_CSV = os.path.join("data", "answer_feedback", "processed", "marks_long.csv")

# Allowed mark values: 0, 0.5, 1.0, 1.5, 2.0
ALLOWED_MARKS = [0.0, 0.5, 1.0, 1.5, 2.0]
LOW_CONF_THRESHOLD = 0.60  # OCR confidence below this triggers a "review" flag


def _load_rubric(rubric_path: str = DEFAULT_RUBRIC_JSON) -> Dict[str, Any]:
    """Loads the rubric JSON."""
    if not os.path.exists(rubric_path):
        raise FileNotFoundError(f"Rubric not found at: {rubric_path}")
    with open(rubric_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_question_means(marks_csv: str = DEFAULT_MARKS_CSV) -> Dict[str, float]:
    """
    Loads historical per-question mean marks from the processed marks CSV.
    Used as a fallback prediction when no ML model is available or answer is empty.
    These are computed on the FULL training set — in production serving, this is
    acceptable as we are not cross-validating; we are using all historical data.
    """
    if not os.path.exists(marks_csv):
        return {}
    df = pd.read_csv(marks_csv)
    return df.groupby("qid")["mark"].mean().to_dict()


def snap_to_allowed(mark: float, max_mark: float = 2.0) -> float:
    """
    Rounds predicted mark to the nearest allowed value {0, 0.5, 1.0, 1.5, 2.0}
    and clips to [0, max_mark].
    """
    clipped = max(0.0, min(mark, max_mark))
    return min(ALLOWED_MARKS, key=lambda v: abs(v - clipped))


def rubric_score(answer_text: Optional[str], rubric_item: Dict[str, Any]) -> Optional[float]:
    """
    Computes a rubric-based predicted mark using keyword coverage.
    Maps coverage fraction → predicted mark in allowed steps.

    - 0 points matched  → 0.0
    - partial coverage  → proportional (rounded to nearest 0.5)
    - all points matched → max_mark

    Returns NaN for missing/empty answers — never returns zero for missing answers.
    """
    max_mark = float(rubric_item.get("max_mark", 2.0))
    if not answer_text or pd.isna(answer_text) or str(answer_text).strip() == "":
        return np.nan  # Missing answer: explicitly unknown, not zero

    coverage, _ = compute_rubric_coverage(answer_text, rubric_item)
    raw = coverage * max_mark
    return snap_to_allowed(raw, max_mark)


class AnswerScorer:
    """
    Main inference class: wraps rubric-based scoring + optional trained champion model.

    Prediction hierarchy:
      1. If answer is missing → NaN (flag for review, NOT zero)
      2. If trained model is loaded → use model features
      3. If no model → use rubric coverage directly
      4. If rubric gives 0 and no model → fall back to question_mean_mark

    Usage:
        scorer = AnswerScorer()
        result = scorer.predict(qid="Q21", answer_text="labeled data training")
    """

    def __init__(
        self,
        rubric_path: str = DEFAULT_RUBRIC_JSON,
        model_path: str = DEFAULT_MODEL_PATH,
        marks_csv: str = DEFAULT_MARKS_CSV,
    ):
        self.rubric = _load_rubric(rubric_path)
        self.question_means = _load_question_means(marks_csv)

        # TF-IDF engine fitted strictly on reference texts — no leakage
        self.tfidf_engine = TfidfSimilarityEngine(rubric=self.rubric)

        # Optionally load the trained champion model.
        # NOTE: The joblib pickle requires the model class to be importable.
        # Import the model classes before loading to register them in sys.modules.
        self.model = None
        if os.path.exists(model_path):
            try:
                from src.answer_feedback.train import (  # noqa: F401 — registers class for pickle
                    BaselineQuestionMeanModel,
                    BaselineTfidfRidgeModel,
                    CandidateGradientBoostingModel,
                )
                self.model = joblib.load(model_path)
                print(f"[INFO] Loaded champion model from {model_path}")
            except Exception as e:
                print(f"[WARN] Could not load model from {model_path}: {e}")

    def predict(
        self,
        qid: str,
        answer_text: Optional[str],
        ocr_conf: Optional[float] = None,
        student_id: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Predicts a mark for a single student-question pair.

        Returns a dict with:
          - predicted_mark:      float in {0, 0.5, 1.0, 1.5, 2.0} or NaN if unattempted
          - prediction_method:   "model" | "rubric_coverage" | "question_mean" | "unattempted"
          - rubric_score:        float — raw rule-based rubric match score
          - tfidf_cos:           float — cosine similarity to reference answer
          - coverage:            float — fraction of rubric points matched
          - requires_review:     bool — True if answer is missing or low OCR confidence
          - ocr_conf:            float | None — forwarded from upstream OCR
          - qid:                 str
        """
        q_data = self.rubric.get(qid)
        if q_data is None:
            return {
                "qid": qid,
                "predicted_mark": np.nan,
                "prediction_method": "unknown_qid",
                "error": f"QID {qid} not found in rubric",
                "requires_review": True,
            }

        max_mark = float(q_data.get("max_mark", 2.0))
        is_missing = (
            answer_text is None
            or (isinstance(answer_text, float) and np.isnan(answer_text))
            or str(answer_text).strip() == ""
        )

        # Rule 1: Missing answer → do NOT assign zero — flag for review
        if is_missing:
            return {
                "qid": qid,
                "predicted_mark": np.nan,
                "prediction_method": "unattempted",
                "rubric_score": np.nan,
                "tfidf_cos": 0.0,
                "coverage": 0.0,
                "max_point_hit": 0,
                "requires_review": True,
                "ocr_conf": ocr_conf,
            }

        # Always compute rubric features (consistent with training pipeline)
        coverage, max_point_hit = compute_rubric_coverage(answer_text, q_data)
        tfidf_cos = self.tfidf_engine.compute_similarity(qid, answer_text)
        rule_score = rubric_score(answer_text, q_data)

        # OCR confidence flag — below threshold needs human review
        low_conf = (
            ocr_conf is not None
            and not (isinstance(ocr_conf, float) and np.isnan(ocr_conf))
            and float(ocr_conf) < LOW_CONF_THRESHOLD
        )
        requires_review = bool(low_conf)

        # Prediction method selection
        if self.model is not None:
            # Use trained model — build feature row using same pipeline as training
            feat_row = extract_features_for_pair(
                student_id=student_id,
                qid=qid,
                answer_text=answer_text,
                ocr_conf=ocr_conf,
                rubric=self.rubric,
                tfidf_engine=self.tfidf_engine,
                question_means=self.question_means,
                true_mark=None,  # NEVER pass true mark as a feature
            )
            feat_df = pd.DataFrame([feat_row])
            try:
                raw_pred = float(self.model.predict(feat_df)[0])
                predicted_mark = snap_to_allowed(raw_pred, max_mark)
                method = "model"
            except Exception:
                # Fallback: rubric score if model predict fails
                predicted_mark = rule_score if not np.isnan(rule_score) else snap_to_allowed(self.question_means.get(qid, 1.0), max_mark)
                method = "rubric_coverage_fallback"
        else:
            # No model — use rubric coverage as primary, question mean as tie-breaker
            if max_point_hit > 0 or coverage > 0:
                predicted_mark = rule_score
                method = "rubric_coverage"
            else:
                q_mean = self.question_means.get(qid, 1.0)
                predicted_mark = snap_to_allowed(q_mean, max_mark)
                method = "question_mean"

        return {
            "qid": qid,
            "predicted_mark": predicted_mark,
            "prediction_method": method,
            "rubric_score": rule_score,
            "tfidf_cos": tfidf_cos,
            "coverage": coverage,
            "max_point_hit": max_point_hit,
            "requires_review": requires_review,
            "ocr_conf": ocr_conf,
        }

    def predict_batch(self, records: list) -> pd.DataFrame:
        """
        Predicts marks for a list of dicts with keys:
            qid, answer_text, ocr_conf (optional), student_id (optional).
        Returns a DataFrame of prediction results.
        """
        results = []
        for rec in records:
            result = self.predict(
                qid=rec.get("qid", ""),
                answer_text=rec.get("answer_text"),
                ocr_conf=rec.get("ocr_conf"),
                student_id=rec.get("student_id", "unknown"),
            )
            results.append(result)
        return pd.DataFrame(results)


if __name__ == "__main__":
    scorer = AnswerScorer()
    test_cases = [
        {"qid": "Q21", "answer_text": "Supervised learning trains a model on labeled data with known inputs and outputs."},
        {"qid": "Q22", "answer_text": "The model is overfitting — it memorized the training data and fails to generalize."},
        {"qid": "Q23", "answer_text": ""},  # Missing — should return NaN
        {"qid": "Q21", "answer_text": "some random irrelevant text with no keywords"},
    ]
    print("=" * 70)
    print("ANSWER SCORER — SMOKE TEST")
    print("=" * 70)
    for tc in test_cases:
        r = scorer.predict(qid=tc["qid"], answer_text=tc["answer_text"])
        print(f"[{r['qid']}] Mark: {r['predicted_mark']} | "
              f"Method: {r['prediction_method']} | "
              f"Coverage: {r.get('coverage', 0):.2f} | "
              f"Review: {r['requires_review']}")
