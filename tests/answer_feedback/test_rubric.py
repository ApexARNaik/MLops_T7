"""
Unit Tests for Answer Feedback Rubric File
"""

import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pytest
from src.answer_feedback.validate_rubric import validate_rubric, RUBRIC_PATH, EXPECTED_QIDS


def test_rubric_file_exists():
    assert os.path.exists(RUBRIC_PATH), f"Rubric file must exist at {RUBRIC_PATH}"


def test_rubric_validation():
    n_questions, n_points = validate_rubric(RUBRIC_PATH)
    assert n_questions == 15, "Expected exactly 15 questions (Q21 to Q35)"
    assert n_points == 39, "Expected 39 rubric points across the 15 questions"


def test_rubric_point_structure():
    with open(RUBRIC_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for qid in EXPECTED_QIDS:
        assert qid in data, f"Missing {qid} in rubric"
        item = data[qid]
        assert item["max_mark"] == 2.0
        assert len(item["reference"]) > 0
        assert 2 <= len(item["points"]) <= 4
        for point in item["points"]:
            assert isinstance(point, list)
            assert len(point) >= 2, f"Point in {qid} should have multiple keyword synonyms"
