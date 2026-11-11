"""
Validation Script for Answer Feedback Rubric
Owner: Member 2 (AV)
Project: AI-Powered Student Study Assistant MLOps Project
Path: src/answer_feedback/validate_rubric.py

Checks:
1. Valid JSON format.
2. All expected question IDs (Q21 to Q35) are present.
3. No missing or unexpected question IDs.
4. Non-empty reference text matching the answer key.
5. Max mark equals 2.0.
6. 2-4 rubric points per question, with each point containing non-empty keyword lists.
7. Prints total questions and rubric points generated.
"""

import os
import json
import csv

RUBRIC_PATH = os.path.join("data", "answer_feedback", "processed", "rubric.json")
ANSWERKEY_PATH = os.path.join("data", "answer_feedback", "raw", "answerkey.txt")
QUESTION_PATH = os.path.join("data", "answer_feedback", "raw", "Question.txt")

EXPECTED_QIDS = [f"Q{i}" for i in range(21, 36)]


def validate_rubric(rubric_path=RUBRIC_PATH):
    print("=" * 80)
    print("VALIDATING RUBRIC FILE:", rubric_path)
    print("=" * 80)

    # 1. Check file existence
    if not os.path.exists(rubric_path):
        raise FileNotFoundError(f"Rubric file not found at: {rubric_path}")

    # 2. Check JSON validity and load
    try:
        with open(rubric_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("[PASS] JSON syntax is valid and loads successfully.")
    except Exception as e:
        raise ValueError(f"[FAIL] Invalid JSON syntax in {rubric_path}: {e}")

    # 3. Load authoritative answer key for comparison
    answerkey_dict = {}
    if os.path.exists(ANSWERKEY_PATH):
        with open(ANSWERKEY_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 3:
                    answerkey_dict[f"Q{row[0]}"] = row[2].strip()

    # 4. Check Question IDs
    actual_qids = list(data.keys())
    missing_qids = [qid for qid in EXPECTED_QIDS if qid not in data]
    unexpected_qids = [qid for qid in actual_qids if qid not in EXPECTED_QIDS]

    if missing_qids:
        raise ValueError(f"[FAIL] Missing question IDs: {missing_qids}")
    if unexpected_qids:
        raise ValueError(f"[FAIL] Unexpected question IDs: {unexpected_qids}")
    print(f"[PASS] All {len(EXPECTED_QIDS)} expected question IDs (Q21 to Q35) are present with no unexpected keys.")

    total_points_count = 0
    validation_errors = []

    print("\nDetailed Per-Question Validation:")
    print(f"{'QID':<6} {'MaxMark':<8} {'Points':<8} {'Reference Length':<18} {'Status':<10}")
    print("-" * 60)

    for qid in EXPECTED_QIDS:
        q_entry = data[qid]

        # Verify entry is a dict
        if not isinstance(q_entry, dict):
            validation_errors.append(f"{qid}: Entry is not a dictionary.")
            continue

        # Check required keys
        for req_key in ["max_mark", "reference", "points"]:
            if req_key not in q_entry:
                validation_errors.append(f"{qid}: Missing required key '{req_key}'.")

        # Check max_mark
        max_mark = q_entry.get("max_mark")
        if max_mark != 2.0 and max_mark != 2:
            validation_errors.append(f"{qid}: max_mark is {max_mark}, expected 2.")

        # Check reference
        ref = q_entry.get("reference", "")
        if not ref or not isinstance(ref, str) or not ref.strip():
            validation_errors.append(f"{qid}: Reference is empty or not a string.")
        elif qid in answerkey_dict and ref.strip() != answerkey_dict[qid]:
            validation_errors.append(
                f"{qid}: Reference text does not match answer key.\n"
                f"  Got:      '{ref.strip()}'\n"
                f"  Expected: '{answerkey_dict[qid]}'"
            )

        # Check points
        pts = q_entry.get("points", [])
        if not isinstance(pts, list) or len(pts) < 2 or len(pts) > 4:
            validation_errors.append(f"{qid}: Has {len(pts)} points, expected 2 to 4 points.")

        for pt_idx, pt in enumerate(pts):
            if not isinstance(pt, list) or len(pt) == 0:
                validation_errors.append(f"{qid} Point #{pt_idx+1}: Not a non-empty list of keywords/phrases.")
            elif any(not isinstance(phrase, str) or not phrase.strip() for phrase in pt):
                validation_errors.append(f"{qid} Point #{pt_idx+1}: Contains blank or non-string phrases.")

        total_points_count += len(pts)
        status = "OK" if not any(err.startswith(qid) for err in validation_errors) else "FAIL"
        print(f"{qid:<6} {str(max_mark):<8} {len(pts):<8} {len(ref):<18} {status:<10}")

    print("-" * 60)
    print(f"Total Questions Validated: {len(EXPECTED_QIDS)}")
    print(f"Total Rubric Points Generated: {total_points_count}")
    print(f"Average Rubric Points per Question: {total_points_count / len(EXPECTED_QIDS):.2f}")

    if validation_errors:
        print("\n[FAIL] Validation Errors Encountered:")
        for err in validation_errors:
            print("  -", err)
        raise ValueError(f"Rubric validation failed with {len(validation_errors)} error(s).")
    else:
        print("\n[SUCCESS] All validation checks passed cleanly with 0 errors!")

    return len(EXPECTED_QIDS), total_points_count


if __name__ == "__main__":
    validate_rubric()

