"""
Tests for predict.py and feedback.py
Owner: Member 2 (AV)
"""
import pytest
import math
import numpy as np
from src.answer_feedback.predict import AnswerScorer, snap_to_allowed, rubric_score
from src.answer_feedback.feedback import generate_feedback, _load_rubric

RUBRIC = {
    "Q21": {
        "qid": "Q21",
        "question": "Define Supervised Learning.",
        "max_mark": 2.0,
        "reference": "A type of ML trained on labeled data with known inputs and outputs.",
        "points": [
            ["labeled data", "labelled data"],
            ["known inputs and outputs", "input-output pairs"]
        ]
    },
    "Q22": {
        "qid": "Q22",
        "question": "What is happening to M1?",
        "max_mark": 2.0,
        "reference": "Overfitting; the model memorized training data but failed to generalize.",
        "points": [
            ["overfitting", "overfit", "high variance"],
            ["memorized the training data", "memorization"],
            ["failed to generalize", "does not generalize"]
        ]
    }
}


# ---------------------------------------------------------------------------
# snap_to_allowed
# ---------------------------------------------------------------------------

def test_snap_to_allowed_exact():
    assert snap_to_allowed(0.0) == 0.0
    assert snap_to_allowed(0.5) == 0.5
    assert snap_to_allowed(1.0) == 1.0
    assert snap_to_allowed(2.0) == 2.0

def test_snap_to_allowed_rounding():
    assert snap_to_allowed(0.3) == 0.5   # rounds up to 0.5
    assert snap_to_allowed(1.1) == 1.0   # rounds down
    assert snap_to_allowed(1.6) == 1.5   # rounds down
    assert snap_to_allowed(1.9) == 2.0   # rounds up

def test_snap_to_allowed_clipping():
    assert snap_to_allowed(-0.5) == 0.0
    assert snap_to_allowed(3.0, max_mark=2.0) == 2.0


# ---------------------------------------------------------------------------
# rubric_score
# ---------------------------------------------------------------------------

def test_rubric_score_full_match():
    # Matches both rubric points for Q21
    ans = "The model is trained on labeled data with known inputs and outputs."
    score = rubric_score(ans, RUBRIC["Q21"])
    assert score == 2.0

def test_rubric_score_missing_answer():
    score = rubric_score("", RUBRIC["Q21"])
    assert score is np.nan or (isinstance(score, float) and math.isnan(score))

def test_rubric_score_no_match():
    score = rubric_score("The sky is blue and grass is green.", RUBRIC["Q21"])
    assert score == 0.0


# ---------------------------------------------------------------------------
# AnswerScorer (no trained model — rubric + question_mean fallback)
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer_no_model(tmp_path, monkeypatch):
    """AnswerScorer with rubric only, no trained model file."""
    import json, os
    rubric_path = tmp_path / "rubric.json"
    rubric_path.write_text(json.dumps(RUBRIC))
    marks_path = tmp_path / "marks.csv"
    marks_path.write_text("student_id,qid,mark\nStudent_1,Q21,2.0\nStudent_1,Q22,1.0\n")
    model_path = tmp_path / "nonexistent_model.joblib"  # doesn't exist
    return AnswerScorer(
        rubric_path=str(rubric_path),
        model_path=str(model_path),
        marks_csv=str(marks_path),
    )

def test_predict_full_match(scorer_no_model):
    r = scorer_no_model.predict("Q21", "trained on labeled data with known inputs and outputs")
    assert r["predicted_mark"] == 2.0
    assert r["requires_review"] is False
    assert r["prediction_method"] in ("rubric_coverage", "model")

def test_predict_missing_answer_is_nan(scorer_no_model):
    r = scorer_no_model.predict("Q21", "")
    assert r["predicted_mark"] is np.nan or (isinstance(r["predicted_mark"], float) and math.isnan(r["predicted_mark"]))
    assert r["requires_review"] is True
    assert r["prediction_method"] == "unattempted"

def test_predict_low_ocr_conf_flags_review(scorer_no_model):
    r = scorer_no_model.predict("Q21", "some answer text", ocr_conf=0.35)
    assert r["requires_review"] is True

def test_predict_unknown_qid(scorer_no_model):
    r = scorer_no_model.predict("Q99", "some text")
    assert r["requires_review"] is True
    assert "error" in r


# ---------------------------------------------------------------------------
# generate_feedback
# ---------------------------------------------------------------------------

def test_feedback_full_match():
    fb = generate_feedback(
        qid="Q21",
        answer_text="trained on labeled data with known inputs and outputs",
        predicted_mark=2.0,
        rubric=RUBRIC,
        ocr_conf=0.85
    )
    assert fb["predicted_mark"] == 2.0
    assert len(fb["matched_points"]) == 2
    assert len(fb["missing_points"]) == 0
    assert "Excellent" in fb["feedback_text"]
    assert fb["requires_review"] is False

def test_feedback_missing_answer():
    fb = generate_feedback(
        qid="Q21",
        answer_text="",
        predicted_mark=None,
        rubric=RUBRIC,
        ocr_conf=None
    )
    assert fb["predicted_mark"] is None
    assert fb["requires_review"] is True
    assert "No answer" in fb["feedback_text"]

def test_feedback_low_confidence_ocr():
    fb = generate_feedback(
        qid="Q22",
        answer_text="the model overfits the training data",
        predicted_mark=1.0,
        rubric=RUBRIC,
        ocr_conf=0.45
    )
    assert fb["requires_review"] is True
    assert fb["ocr_warning"] is not None
    assert "Low OCR confidence" in fb["ocr_warning"]

def test_feedback_partial_match():
    fb = generate_feedback(
        qid="Q22",
        answer_text="the model is overfitting",
        predicted_mark=0.5,
        rubric=RUBRIC,
        ocr_conf=0.75
    )
    assert 0 < fb["n_matched"] < fb["n_total_points"]
    assert len(fb["missing_points"]) > 0

def test_feedback_unknown_qid():
    fb = generate_feedback(
        qid="Q99",
        answer_text="some text",
        predicted_mark=None,
        rubric=RUBRIC,
    )
    assert fb["requires_review"] is True
    assert "error" in fb
