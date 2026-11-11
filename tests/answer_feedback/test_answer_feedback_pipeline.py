"""
Comprehensive PyTest Unit Tests for Answer Feedback Module
Owner: Member 2 (AV)
Path: tests/answer_feedback/test_answer_feedback_pipeline.py

Verifies the 10 core functional and safety requirements:
1. Predicted marks are always between 0 and max_mark.
2. Empty answers require human review.
3. Low OCR confidence below 0.6 requires human review.
4. Answers containing fewer than three words require human review.
5. Unknown question IDs are handled safely (raises ValueError).
6. Rubric coverage works correctly on a manually constructed example.
7. Matched and missing rubric points are correct.
8. The true mark column is never included in the feature list.
9. Model metadata contains a model version.
10. A valid answer returns the expected response structure.
"""

import json
from pathlib import Path
import pytest

from src.answer_feedback.predict import evaluate, load_metadata, snap_to_allowed
from src.answer_feedback.feedback import compute_point_breakdown
from src.answer_feedback.features import compute_rubric_coverage, extract_features_for_pair, TfidfSimilarityEngine

REPO_ROOT = Path(__file__).resolve().parents[2]
META_PATH = REPO_ROOT / "models" / "answer_meta.json"
RUBRIC_PATH = REPO_ROOT / "data" / "answer_feedback" / "processed" / "rubric.json"


# 1. Predicted marks are always between 0 and max_mark
def test_predicted_marks_bounded_between_zero_and_max_mark():
    answers = {
        "Q21": "Supervised learning trains a model on labeled data with known inputs and outputs.",
        "Q22": "This model is experiencing severe overfitting on the training data.",
        "Q23": "Completely off-topic irrelevant text that should receive low or baseline marks.",
        "Q24": "Short.",
    }
    ocr_conf = {"Q21": 0.95, "Q22": 0.85, "Q23": 0.90, "Q24": 0.70}

    response = evaluate(answers, ocr_conf=ocr_conf)
    for result in response["data"]["results"]:
        mark = result["estimated_mark"]
        max_mark = result["max_mark"]
        assert 0.0 <= mark <= max_mark, f"Mark {mark} for {result['qid']} is outside [0, {max_mark}]"

    # Also test extreme raw inputs in snap_to_allowed
    assert snap_to_allowed(-5.0, max_mark=2.0) == 0.0
    assert snap_to_allowed(10.0, max_mark=2.0) == 2.0


# 2. Empty answers require human review
def test_empty_answers_require_human_review():
    empty_cases = ["", "   ", "\n\t  "]
    for empty_text in empty_cases:
        response = evaluate({"Q21": empty_text}, ocr_conf=None)
        result = response["data"]["results"][0]
        assert result["human_review_required"] is True, "Empty answer did not trigger human_review_required"
        assert result["estimated_mark"] == 0.0, "Empty answer must receive 0.0 estimated mark"


# 3. Low OCR confidence below 0.6 requires human review
def test_low_ocr_confidence_requires_human_review():
    answer = "Supervised learning uses labeled dataset with known inputs and outputs."
    # Above threshold: 0.60, 0.75 -> review not triggered by confidence
    resp_high = evaluate({"Q21": answer}, ocr_conf={"Q21": 0.75})
    assert resp_high["data"]["results"][0]["human_review_required"] is False

    # Below threshold: 0.59, 0.30 -> review must be triggered
    for low_conf in (0.59, 0.30, 0.05):
        resp_low = evaluate({"Q21": answer}, ocr_conf={"Q21": low_conf})
        res = resp_low["data"]["results"][0]
        assert res["human_review_required"] is True, f"Confidence {low_conf} did not trigger human_review_required"
        assert res["ocr_confidence"] == low_conf


# 4. Answers containing fewer than three words require human review
def test_short_answers_require_human_review():
    # 1 word and 2 words
    short_cases = ["Yes", "Overfitting model"]
    for short_ans in short_cases:
        resp = evaluate({"Q21": short_ans}, ocr_conf={"Q21": 0.95})
        res = resp["data"]["results"][0]
        assert res["human_review_required"] is True, f"Short answer '{short_ans}' did not trigger human_review_required"


# 5. Unknown question IDs are handled safely
def test_unknown_question_ids_handled_safely():
    with pytest.raises(ValueError) as excinfo:
        evaluate({"Q999": "Some answer text"})
    # Must report unknown question without leaking raw answer text
    assert "Unknown question ID 'Q999'" in str(excinfo.value)
    assert "Some answer text" not in str(excinfo.value)


# 6. Rubric coverage works correctly on a manually constructed example
def test_rubric_coverage_manual_example():
    manual_rubric_item = {
        "max_mark": 2.0,
        "reference": "A model trained on labeled data with known inputs and outputs.",
        "points": [
            ["labeled data", "labelled dataset"],
            ["known inputs and outputs", "input-output pairs"]
        ]
    }
    # Matches point 1 only
    cov1, hit1 = compute_rubric_coverage("We train the model using a labelled dataset.", manual_rubric_item)
    assert cov1 == 0.5
    assert hit1 == 1

    # Matches both points
    cov2, hit2 = compute_rubric_coverage("Trained on labeled data with input-output pairs.", manual_rubric_item)
    assert cov2 == 1.0
    assert hit2 == 1

    # Matches none
    cov0, hit0 = compute_rubric_coverage("Linear algebra matrices and eigenvalues.", manual_rubric_item)
    assert cov0 == 0.0
    assert hit0 == 0


# 7. Matched and missing rubric points are correct
def test_matched_and_missing_points_are_correct():
    manual_rubric_item = {
        "points": [
            ["overfitting", "overfit"],
            ["memorized the training data", "memorized training data"],
            ["failed to generalize", "does not generalize"]
        ]
    }
    answer = "The model is overfit and failed to generalize to unseen test data."
    matched, missing = compute_point_breakdown(answer, manual_rubric_item)

    assert "overfitting" in matched
    assert "failed to generalize" in matched
    assert "memorized the training data" in missing
    assert len(matched) == 2
    assert len(missing) == 1


# 8. The true mark column is never included in the feature list
def test_true_mark_never_in_feature_list():
    meta = load_metadata()
    features = meta.get("features", [])
    assert "mark" not in features, "Direct 'mark' found in model feature list!"
    assert "true_mark" not in features, "Direct 'true_mark' found in model feature list!"

    # Verify extract_features_for_pair does not leak true_mark into feature calculation
    dummy_rubric = {
        "Q21": {
            "max_mark": 2.0,
            "reference": "Supervised learning reference.",
            "points": [["supervised learning"]]
        }
    }
    engine = TfidfSimilarityEngine(dummy_rubric)
    feat_record = extract_features_for_pair(
        student_id="test_student",
        qid="Q21",
        answer_text="supervised learning concept",
        ocr_conf=0.9,
        rubric=dummy_rubric,
        tfidf_engine=engine,
        question_means={"Q21": 1.5},
        true_mark=None
    )
    # The feature dict keys representing the model input vector:
    input_feature_keys = [k for k in feat_record.keys() if k != "mark"]
    assert "mark" not in input_feature_keys
    assert "true_mark" not in input_feature_keys


# 9. Model metadata contains a model version
def test_model_metadata_contains_model_version():
    meta = load_metadata()
    assert "model_version" in meta, "Missing 'model_version' in answer_meta.json"
    version = str(meta["model_version"]).strip()
    assert len(version) > 0, "model_version cannot be empty"
    assert "model_name" in meta, "Missing 'model_name' in answer_meta.json"


# 10. A valid answer returns the expected response structure
def test_valid_answer_returns_expected_structure():
    answers = {"Q21": "Supervised learning trains a model on labeled data with known inputs and outputs."}
    response = evaluate(answers, ocr_conf={"Q21": 0.90})

    # Envelope validation
    assert "model_name" in response
    assert "model_version" in response
    assert "warnings" in response
    assert isinstance(response["warnings"], list)
    assert "data" in response
    assert "results" in response["data"]

    # Results item validation
    results = response["data"]["results"]
    assert len(results) == 1
    res = results[0]
    expected_fields = [
        "qid",
        "estimated_mark",
        "max_mark",
        "approved_mark",
        "status",
        "human_review_required",
        "ocr_confidence",
        "matched_points",
        "missing_points",
        "deduction_rationale",
        "improvement_suggestion",
        "evidence_refs",
        "feedback_text",
    ]
    for field in expected_fields:
        assert field in res, f"Expected field '{field}' missing from result entry"

    assert res["qid"] == "Q21"
    assert res["status"] == "pending_teacher_review"
    assert res["approved_mark"] is None
    assert isinstance(res["evidence_refs"], list)
    assert len(res["evidence_refs"]) > 0
    assert any("answerkey.txt" in ref for ref in res["evidence_refs"])

