"""
Production Feedback Generation Module for Answer-Sheet Scoring
Owner: Member 2 (AV)
Project: AI-Powered Student Study Assistant MLOps Project
Path: src/answer_feedback/feedback.py

Generates deterministic, template-based feedback for student short-answer responses
using rubric keyword point matching without LLMs or external APIs.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.answer_feedback.features import normalize_text


def compute_point_breakdown(
    answer_text: Optional[str],
    rubric_item: Dict[str, Any],
) -> tuple[List[str], List[str]]:
    """
    Computes matched_points and missing_points based on actual rubric keyword variants.
    
    Matching rules:
      - Exact normalized subphrase in answer text, OR
      - >= 60% token overlap with any variant list of the rubric point.
      
    Returns:
      (matched_points, missing_points):
        matched_points: list of descriptive concept strings that were hit
        missing_points: list of descriptive concept strings that were missed
    """
    if answer_text is None or not str(answer_text).strip():
        # Completely missing answer: all rubric points are missed
        all_points = [
            variants[0] if variants else "key concept"
            for variants in rubric_item.get("points", [])
        ]
        return [], all_points

    norm_answer = normalize_text(answer_text)
    student_tokens = set(norm_answer.split()) if norm_answer else set()

    matched_points: List[str] = []
    missing_points: List[str] = []

    for point_variants in rubric_item.get("points", []):
        if not point_variants:
            continue

        concept_label = point_variants[0]
        matched = False

        for variant in point_variants:
            var_norm = normalize_text(variant)
            if not var_norm:
                continue

            # Exact phrase match in normalized answer
            if var_norm in norm_answer:
                matched = True
                break

            # >= 60% token overlap
            var_tokens = set(var_norm.split())
            if var_tokens:
                overlap = len(student_tokens.intersection(var_tokens))
                if overlap / len(var_tokens) >= 0.60:
                    matched = True
                    break

        if matched:
            matched_points.append(concept_label)
        else:
            missing_points.append(concept_label)

    return matched_points, missing_points


def build_deterministic_feedback(
    qid: str,
    answer_text: Optional[str],
    estimated_mark: float,
    max_mark: float,
    matched_points: List[str],
    missing_points: List[str],
    rubric_item: Dict[str, Any],
    requires_review: bool,
    review_reasons: List[str],
) -> Dict[str, Any]:
    """
    Constructs deterministic, template-based feedback explaining points matched,
    deductions, and improvement suggestions without any LLM or external API.
    """
    reference = rubric_item.get("reference", "")
    total_points = len(rubric_item.get("points", []))
    n_matched = len(matched_points)

    # 1. Deduction rationale
    if not answer_text or not str(answer_text).strip():
        deduction_rationale = (
            "No answer was detected. Full marks deducted pending human verification."
        )
    elif len(answer_text.split()) < 3:
        deduction_rationale = (
            f"The response is only {len(answer_text.split())} word(s) long and lacked key explanatory concepts."
        )
    elif missing_points:
        missing_str = ", ".join(f"'{p}'" for p in missing_points)
        deduction_rationale = (
            f"Deducted for omission of key expected concept(s): {missing_str}."
        )
    else:
        deduction_rationale = "No deductions. All core rubric concepts were addressed."

    # 2. Improvement suggestion
    if missing_points:
        missing_str = ", ".join(f"'{p}'" for p in missing_points)
        improvement_suggestion = (
            f"To achieve full marks, ensure your answer explicitly discusses {missing_str} as detailed in the answer key."
        )
    else:
        improvement_suggestion = (
            "Excellent answer! Keep providing clear, concise definitions matching key terminology."
        )

    # 3. Overall feedback summary text
    feedback_lines = []
    if requires_review:
        reasons_str = "; ".join(review_reasons)
        feedback_lines.append(f"[Teacher Review Required: {reasons_str}]")

    feedback_lines.append(f"Estimated Mark: {estimated_mark:.1f} / {max_mark:.1f}")

    if not answer_text or not str(answer_text).strip():
        feedback_lines.append("No answer detected.")
    elif n_matched == total_points and total_points > 0:
        feedback_lines.append("Excellent! All key rubric criteria were satisfied.")
    elif n_matched > 0:
        feedback_lines.append(
            f"Covered {n_matched} of {total_points} expected criteria. Addressed: {', '.join(matched_points)}."
        )
    else:
        feedback_lines.append("Did not clearly address the expected key concepts.")

    if missing_points:
        feedback_lines.append(f"Missing elements: {', '.join(missing_points)}.")

    feedback_lines.append(f"Reference concept: {reference}")
    feedback_text = " ".join(feedback_lines)

    return {
        "deduction_rationale": deduction_rationale,
        "improvement_suggestion": improvement_suggestion,
        "feedback_text": feedback_text,
    }


# Backwards compatibility functions for tests
def _load_rubric(rubric_path=None):
    from src.answer_feedback.predict import load_rubric
    return load_rubric()


def generate_feedback(qid: str, answer_text: Optional[str], predicted_mark: Optional[float], rubric: Dict[str, Any], ocr_conf: Optional[float] = None):
    rubric_item = rubric.get(qid)
    if not rubric_item:
        return {"qid": qid, "error": f"Unknown question {qid}", "requires_review": True}
    max_mark = float(rubric_item.get("max_mark", 2.0))
    matched_pts, missing_pts = compute_point_breakdown(answer_text, rubric_item)
    requires_review = bool(not answer_text or (ocr_conf is not None and ocr_conf < 0.60))
    review_reasons = []
    if not answer_text:
        review_reasons.append("Answer is empty")
    if ocr_conf is not None and ocr_conf < 0.60:
        review_reasons.append(f"Low OCR confidence ({ocr_conf:.2f} < 0.60)")
    
    details = build_deterministic_feedback(
        qid=qid,
        answer_text=answer_text,
        estimated_mark=predicted_mark or 0.0,
        max_mark=max_mark,
        matched_points=matched_pts,
        missing_points=missing_pts,
        rubric_item=rubric_item,
        requires_review=requires_review,
        review_reasons=review_reasons,
    )
    return {
        "qid": qid,
        "predicted_mark": predicted_mark,
        "matched_points": [{"concept": p} for p in matched_pts],
        "missing_points": [{"concept": p} for p in missing_pts],
        "n_matched": len(matched_pts),
        "n_total_points": len(rubric_item.get("points", [])),
        "feedback_text": details["feedback_text"],
        "requires_review": requires_review,
        "ocr_warning": f"Low OCR confidence ({ocr_conf:.2f})" if (ocr_conf is not None and ocr_conf < 0.60) else None,
    }

