"""
Unit Tests for Answer Feedback Baseline Pipeline
"""

import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src.answer_feedback.data import load_raw_marks, build_long_format_marks, PROCESSED_LONG_PATH
from src.answer_feedback.train import QuestionMeanBaseline, evaluate_predictions


def test_long_format_data_integrity():
    df = pd.read_csv(PROCESSED_LONG_PATH)
    assert len(df) == 568, f"Expected 568 samples, got {len(df)}"
    assert df["student_id"].nunique() == 50, f"Expected 50 distinct students, got {df['student_id'].nunique()}"
    assert set(df["qid"].unique()) == {f"Q{i}" for i in range(21, 36)}
    assert df["mark"].isnull().sum() == 0, "No null marks allowed in long format"
    assert df["mark"].min() >= 0.0, "Marks must be non-negative"


def test_question_mean_baseline_fit_predict():
    dummy_X = pd.DataFrame({
        "qid": ["Q21", "Q21", "Q22", "Q22"]
    })
    dummy_y = pd.Series([2.0, 1.0, 1.0, 0.0])

    model = QuestionMeanBaseline(step=0.5, max_mark=2.0)
    model.fit(dummy_X, dummy_y)

    # Q21 mean is 1.5, Q22 mean is 0.5
    assert model.question_means_["Q21"] == 1.5
    assert model.question_means_["Q22"] == 0.5
    assert model.global_mean_ == 1.0

    test_X = pd.DataFrame({"qid": ["Q21", "Q22", "Q99"]})
    preds = model.predict(test_X, round_to_step=True)
    assert preds[0] == 1.5
    assert preds[1] == 0.5
    assert preds[2] == 1.0  # fallback to global mean


def test_group_kfold_no_student_leakage():
    df = pd.read_csv(PROCESSED_LONG_PATH)
    gkf = GroupKFold(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["student_id"]), 1):
        train_students = set(df.iloc[train_idx]["student_id"])
        test_students = set(df.iloc[test_idx]["student_id"])
        assert len(train_students.intersection(test_students)) == 0, f"Student leakage in fold {fold}"
        assert len(test_students) == 10
        assert len(train_students) == 40

