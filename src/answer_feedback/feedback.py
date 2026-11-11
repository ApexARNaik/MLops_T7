"""
Feedback Generation Module for Answer Feedback
Owner: Member 2 (AV)
Project: AI-Powered Student Study Assistant MLOps Project
Path: src/answer_feedback/feedback.py

Given a student's answer_text, predicted_mark, and qid, generates structured
textual feedback that:
  - Explains which rubric points were matched correctly.
  - Identifies which key concepts were missing from the answer.
  - States the predicted mark and maximum mark.
  - Flags low OCR confidence answers for human review.
  - Never fabricates rubric content or invents marks.
  - Never references Corrected_Pdf.

Output is JSON-serializable and can be served via the FastAPI endpoint.
"""

import os
import json
import math
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List

from src.answer_feedback.features import (
    normalize_text,
    compute_rubric_coverage,
    match_rubric_point,
)

# Default rubric path
DEFAULT_RUBRIC_JSON = os.path.join("data", "answer_feedback", "processed", "rubric.json")
LOW_CONF_THRESHOLD = 0.60
MIN_READABLE_WORDS = 3  # Answers shorter than this are likely OCR noise


def _load_rubric(rubric_path: str = DEFAULT_RUBRIC_JSON) -> Dict[str, Any]:
    if not os.path.exists(rubric_path):
        raise FileNotFoundError(f"Rubric not found at: {rubric_path}")
    with open(rubric_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_matched_points(answer_text: str, rubric_item: Dict[str, Any]) -> List[Dict]:
    """
    Returns a per-rubric-point breakdown showing which keyword groups were hit or missed.
    Each entry carries:
      - matched: bool
      - matched_phrase: str (best-matching variant, or None)
      - all_variants: list[str] — representative alternatives for feedback display
    """
    norm_answer = normalize_text(answer_text)
    student_tokens = set(norm_answer.split()) if norm_answer else set()
    breakdown = []

    for point_variants in rubric_item.get("points", []):
        matched = False
        matched_phrase = None

        for variant in point_variants:
            var_norm = normalize_text(variant)
            if not var_norm:
                continue
            var_tokens = set(var_norm.split())

            # Exact phrase match
            if var_norm in norm_answer:
                matched = True
                matched_phrase = variant
                break

            # 60% token overlap
            if var_tokens and len(student_tokens.intersection(var_tokens)) / len(var_tokens) >= 0.60:
                matched = True
                matched_phrase = variant
                break

        # For feedback, show first 2 representative alternatives
        display_variants = point_variants[:2]
        breakdown.append({
            "matched": matched,
            "matched_phrase": matched_phrase,
            "representative_keywords": display_variants,
        })

    return breakdown


def generate_feedback(
    qid: str,
    answer_text: Optional[str],
    predicted_mark: Optional[float],
    rubric: Dict[str, Any],
    ocr_conf: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generates structured feedback for a single student-question pair.

    Parameters:
        qid:             Question identifier (e.g. "Q21")
        answer_text:     Extracted student answer (from OCR or manual transcription)
        predicted_mark:  Predicted mark (from predict.py scorer)
        rubric:          Full rubric dict loaded from rubric.json
        ocr_conf:        OCR confidence score (or None if not from OCR pipeline)

    Returns a dict with:
        qid, question, predicted_mark, max_mark, reference_answer,
        matched_points (list), missing_points (list),
        feedback_text (str), requires_review (bool), ocr_warning (str | None)
    """
    q_data = rubric.get(qid)
    if q_data is None:
        return {
            "qid": qid,
            "error": f"QID {qid} not found in rubric.",
            "feedback_text": "Unable to generate feedback: question not found in rubric.",
            "requires_review": True,
        }

    max_mark = float(q_data.get("max_mark", 2.0))
    reference = q_data.get("reference", "")
    question_text = q_data.get("question", qid)

    # OCR warning construction
    ocr_warning = None
    if ocr_conf is not None and not (isinstance(ocr_conf, float) and np.isnan(ocr_conf)):
        if float(ocr_conf) < LOW_CONF_THRESHOLD:
            ocr_warning = (
                f"⚠️ Low OCR confidence ({float(ocr_conf):.2f}). "
                "The extracted text may contain recognition errors. "
                "Human review of the original answer sheet is recommended."
            )

    # Case 1: Missing or empty answer
    is_missing = (
        answer_text is None
        or (isinstance(answer_text, float) and np.isnan(answer_text))
        or str(answer_text).strip() == ""
    )
    if is_missing:
        return {
            "qid": qid,
            "question": question_text,
            "predicted_mark": None,
            "max_mark": max_mark,
            "reference_answer": reference,
            "matched_points": [],
            "missing_points": [pt[:2] for pt in q_data.get("points", [])],
            "feedback_text": (
                "No answer was detected for this question. "
                "If the student answered this question, the handwritten text was not "
                "successfully extracted by OCR. Please review the original answer sheet."
            ),
            "requires_review": True,
            "ocr_warning": ocr_warning,
        }

    # Case 2: Answer too short (likely OCR noise)
    word_count = len(normalize_text(answer_text).split())
    if word_count < MIN_READABLE_WORDS:
        return {
            "qid": qid,
            "question": question_text,
            "predicted_mark": predicted_mark,
            "max_mark": max_mark,
            "reference_answer": reference,
            "matched_points": [],
            "missing_points": [pt[:2] for pt in q_data.get("points", [])],
            "feedback_text": (
                f"The extracted answer is very short ({word_count} word(s)) and may "
                "represent OCR noise rather than the student's actual response. "
                "Human verification of the original answer sheet is recommended."
            ),
            "requires_review": True,
            "ocr_warning": ocr_warning,
        }

    # Case 3: Normal feedback generation
    point_breakdown = _get_matched_points(answer_text, q_data)

    matched_points = []
    missing_points = []
    for pb in point_breakdown:
        if pb["matched"]:
            matched_points.append({
                "concept": pb["representative_keywords"][0],
                "matched_as": pb["matched_phrase"],
            })
        else:
            missing_points.append({
                "concept": pb["representative_keywords"][0],
                "hint": f"Look for keywords such as: {', '.join(pb['representative_keywords'])}",
            })

    n_matched = len(matched_points)
    n_total = len(point_breakdown)

    # Build human-readable feedback text
    lines = []
    if predicted_mark is not None and not (isinstance(predicted_mark, float) and np.isnan(predicted_mark)):
        lines.append(f"**Predicted mark: {predicted_mark:.1f} / {max_mark:.1f}**")
    else:
        lines.append(f"**Mark: requires review / {max_mark:.1f}**")

    if n_matched == 0:
        lines.append(
            "\nYour answer did not clearly match the key concepts expected for this question."
        )
    elif n_matched == n_total:
        lines.append(
            f"\nExcellent! Your answer addressed all {n_total} key concept(s) expected."
        )
    else:
        lines.append(
            f"\nYour answer addressed {n_matched} of {n_total} key concept(s)."
        )

    if matched_points:
        lines.append("\n✅ **Concepts you covered:**")
        for mp in matched_points:
            lines.append(f"  • {mp['concept']}")

    if missing_points:
        lines.append("\n❌ **Concepts not found in your answer:**")
        for mp in missing_points:
            lines.append(f"  • {mp['concept']} — {mp['hint']}")

    lines.append(f"\n📖 **Reference answer:** {reference}")

    if ocr_warning:
        lines.append(f"\n{ocr_warning}")

    feedback_text = "\n".join(lines)

    return {
        "qid": qid,
        "question": question_text,
        "predicted_mark": predicted_mark,
        "max_mark": max_mark,
        "reference_answer": reference,
        "matched_points": matched_points,
        "missing_points": missing_points,
        "n_matched": n_matched,
        "n_total_points": n_total,
        "feedback_text": feedback_text,
        "requires_review": bool(ocr_warning is not None),
        "ocr_warning": ocr_warning,
    }


def generate_feedback_for_student(
    student_answers: List[Dict],
    rubric: Dict[str, Any],
) -> List[Dict]:
    """
    Generates feedback for all of a student's short-answer responses.

    Parameters:
        student_answers: list of dicts with keys:
            qid, answer_text, predicted_mark, ocr_conf (optional)
        rubric: loaded rubric.json dict

    Returns list of feedback dicts, one per question attempt.
    """
    return [
        generate_feedback(
            qid=ans.get("qid", ""),
            answer_text=ans.get("answer_text"),
            predicted_mark=ans.get("predicted_mark"),
            rubric=rubric,
            ocr_conf=ans.get("ocr_conf"),
        )
        for ans in student_answers
    ]


if __name__ == "__main__":
    rubric = _load_rubric()

    test_cases = [
        {
            "qid": "Q21",
            "answer_text": "Supervised learning trains a model on labeled data with known inputs and outputs.",
            "predicted_mark": 2.0,
            "ocr_conf": 0.82,
        },
        {
            "qid": "Q22",
            "answer_text": "high accuracy on training but low on test",
            "predicted_mark": 0.5,
            "ocr_conf": 0.55,
        },
        {
            "qid": "Q23",
            "answer_text": "",
            "predicted_mark": None,
            "ocr_conf": None,
        },
    ]

    print("=" * 70)
    print("FEEDBACK GENERATOR — SMOKE TEST")
    print("=" * 70)
    for tc in test_cases:
        fb = generate_feedback(
            qid=tc["qid"],
            answer_text=tc["answer_text"],
            predicted_mark=tc["predicted_mark"],
            rubric=rubric,
            ocr_conf=tc["ocr_conf"],
        )
        print(f"\n--- [{fb['qid']}] ---")
        print(fb["feedback_text"])
        print(f"Requires review: {fb['requires_review']}")
