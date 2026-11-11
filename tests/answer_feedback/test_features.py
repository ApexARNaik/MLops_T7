import pytest
import os
import numpy as np
import pandas as pd
from src.answer_feedback.features import (
    normalize_text,
    compute_word_counts,
    match_rubric_point,
    compute_rubric_coverage,
    TfidfSimilarityEngine,
    compute_question_means,
    extract_features_for_pair
)

def test_normalize_text():
    raw = "Supervised Learning, with (labeled) data! \n\n"
    assert normalize_text(raw) == "supervised learning with labeled data"
    assert normalize_text(None) == ""
    assert normalize_text(np.nan) == ""

def test_word_counts():
    text = "Supervised learning uses labeled training datasets."
    n_w, log_w = compute_word_counts(text)
    assert n_w == 6
    assert abs(log_w - np.log1p(6)) < 1e-4

    n_w_empty, log_w_empty = compute_word_counts("")
    assert n_w_empty == 0
    assert log_w_empty == 0.0

def test_rubric_matching_and_coverage():
    rubric_item = {
        "max_mark": 2.0,
        "reference": "Model trained on labeled data with known inputs and outputs.",
        "points": [
            ["labeled data", "labelled dataset"],
            ["known inputs and outputs", "known input output"]
        ]
    }
    answer = "The system is trained on labeled data to map inputs to outputs."
    cov, hit = compute_rubric_coverage(answer, rubric_item)
    assert cov == 0.5
    assert hit == 1

def test_tfidf_similarity_engine():
    rubric = {
        "Q21": {
            "reference": "Supervised learning models use labeled data.",
            "points": [["labeled data"]]
        }
    }
    engine = TfidfSimilarityEngine(rubric)
    sim = engine.compute_similarity("Q21", "This model relies on labeled data for learning.")
    assert sim > 0.0
    assert sim <= 1.0

    # Non-matching text
    sim_zero = engine.compute_similarity("Q21", "Completely irrelevant bananas and apples.")
    assert sim_zero == 0.0

def test_extract_features_no_leakage():
    rubric = {
        "Q21": {
            "max_mark": 2.0,
            "reference": "Supervised learning is trained on labeled data.",
            "points": [["labeled data"]]
        }
    }
    engine = TfidfSimilarityEngine(rubric)
    means = {"Q21": 1.5}
    feat = extract_features_for_pair(
        student_id="Student_1",
        qid="Q21",
        answer_text="trained on labeled data",
        ocr_conf=0.85,
        rubric=rubric,
        tfidf_engine=engine,
        question_means=means,
        true_mark=2.0
    )
    assert feat["tfidf_cos"] > 0.0
    assert feat["coverage"] == 1.0
    assert feat["max_point_hit"] == 1
    assert feat["n_words"] == 4
    assert feat["ocr_conf"] == 0.85
    assert feat["question_mean_mark"] == 1.5
    assert feat["max_mark"] == 2.0
    assert feat["is_missing"] == 0
    assert feat["mark"] == 2.0
