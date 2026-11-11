"""Unit tests for Feature A: Data Engineering (src/hobby_learning/data.py)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.hobby_learning.data import (
    clean_and_validate_data,
    engineer_historical_features,
    normalize_skills,
    process_dataset,
)


@pytest.fixture
def sample_interactions() -> pd.DataFrame:
    """Fixture providing sample interactions for testing."""
    data = {
        "order_id": [10, 20, 30, 40, 50],
        "user_id": ["u1", "u1", "u1", "u2", "u2"],
        "problem_id": ["p1", "p2", "p3", "p4", "p5"],
        "correct": [1, 0, 1, 1, 0],
        "skill_id": ["1_13", "1_13", None, "85", "163"],
        "skill_name": ["Box and Whisker", "Box and Whisker", None, "Absolute Value", "Absolute Value"],
        "attempt_count": [1, 2, 3, 1, 4],
        "hint_count": [0, 1, 2, 0, 3],
        "original": [1, 1, 1, 1, 1],
        "ms_first_response": [1000, 2000, 3000, 1500, 2500],
    }
    return pd.DataFrame(data)


def test_historical_correctness_excludes_current_event():
    """Test 1: Historical correctness excludes the current event."""
    df = pd.DataFrame({
        "order_id": [1, 2, 3],
        "user_id": ["u1", "u1", "u1"],
        "problem_id": ["p1", "p2", "p3"],
        "correct": [1, 0, 1],
        "skill_id": ["1", "1", "1"],
        "skill_name": ["SkillA", "SkillA", "SkillA"],
        "attempt_count": [1, 2, 3],
        "hint_count": [0, 1, 2],
    })
    cleaned, _ = clean_and_validate_data(df)
    norm, _, _ = normalize_skills(cleaned)
    events = engineer_historical_features(norm)

    # First event: count is 0, accuracy is NaN
    assert events.loc[0, "user_prev_count"] == 0
    assert np.isnan(events.loc[0, "user_prev_accuracy"])

    # Second event: only sees event 1 (correct = 1) -> acc = 1.0
    assert events.loc[1, "user_prev_count"] == 1
    assert events.loc[1, "user_prev_accuracy"] == 1.0

    # Third event: sees event 1 and 2 (1 correct out of 2) -> acc = 0.5
    assert events.loc[2, "user_prev_count"] == 2
    assert events.loc[2, "user_prev_accuracy"] == 0.5


def test_future_events_do_not_influence_previous_features():
    """Test 2: Future events do not influence previous features."""
    df_short = pd.DataFrame({
        "order_id": [1, 2],
        "user_id": ["u1", "u1"],
        "problem_id": ["p1", "p2"],
        "correct": [1, 0],
        "skill_id": ["1", "1"],
        "skill_name": ["SkillA", "SkillA"],
        "attempt_count": [1, 2],
        "hint_count": [0, 1],
    })
    df_long = pd.concat([
        df_short,
        pd.DataFrame({
            "order_id": [3, 4],
            "user_id": ["u1", "u1"],
            "problem_id": ["p3", "p4"],
            "correct": [1, 1],
            "skill_id": ["1", "1"],
            "skill_name": ["SkillA", "SkillA"],
            "attempt_count": [1, 1],
            "hint_count": [0, 0],
        }),
    ], ignore_index=True)

    cleaned_short, _ = clean_and_validate_data(df_short)
    norm_short, _, _ = normalize_skills(cleaned_short)
    res_short = engineer_historical_features(norm_short)

    cleaned_long, _ = clean_and_validate_data(df_long)
    norm_long, _, _ = normalize_skills(cleaned_long)
    res_long = engineer_historical_features(norm_long)

    # Event 0 and Event 1 metrics must be strictly identical regardless of future rows
    pd.testing.assert_series_equal(res_short["user_prev_count"], res_long["user_prev_count"].iloc[:2])
    pd.testing.assert_series_equal(res_short["user_prev_accuracy"], res_long["user_prev_accuracy"].iloc[:2])
    pd.testing.assert_series_equal(res_short["user_prev_attempts_mean"], res_long["user_prev_attempts_mean"].iloc[:2])


def test_ordering_uses_numeric_order_id():
    """Test 3: Ordering uses numeric order_id (e.g. 10 comes before 9 if sorted as string, but numeric 9 < 10)."""
    # If sorted alphabetically, "100" < "20" < "9"
    df = pd.DataFrame({
        "order_id": ["100", "20", "9"],
        "user_id": ["u1", "u1", "u1"],
        "problem_id": ["p1", "p2", "p3"],
        "correct": [1, 0, 1],
        "skill_id": ["1", "1", "1"],
        "skill_name": ["A", "A", "A"],
        "attempt_count": [1, 1, 1],
        "hint_count": [0, 0, 0],
    })
    cleaned, _ = clean_and_validate_data(df)
    norm, _, _ = normalize_skills(cleaned)
    res = engineer_historical_features(norm)

    # Numeric order: 9, 20, 100
    assert list(res["order_id"]) == [9, 20, 100]
    # At order_id=20, previous was order_id=9 (correct=1) -> acc = 1.0
    row_20 = res[res["order_id"] == 20].iloc[0]
    assert row_20["user_prev_accuracy"] == 1.0


def test_untagged_interactions_contribute_to_learner_history():
    """Test 4: Untagged interactions contribute to learner history."""
    df = pd.DataFrame({
        "order_id": [1, 2, 3],
        "user_id": ["u1", "u1", "u1"],
        "problem_id": ["p1", "p2", "p3"],
        "correct": [1, 0, 1],
        "skill_id": [None, "1", "1"],
        "skill_name": [None, "SkillA", "SkillA"],
        "attempt_count": [1, 2, 3],
        "hint_count": [0, 1, 2],
    })
    cleaned, _ = clean_and_validate_data(df)
    norm, _, _ = normalize_skills(cleaned)
    res = engineer_historical_features(norm)

    # Event 2 has 1 previous interaction in learner-wide history (event 1, which had no skill)
    assert res.loc[1, "user_prev_count"] == 1
    assert res.loc[1, "user_prev_accuracy"] == 1.0
    assert res.loc[2, "user_prev_count"] == 2
    assert res.loc[2, "user_prev_accuracy"] == 0.5


def test_untagged_interactions_do_not_share_common_skill_history():
    """Test 5: Untagged interactions do not share a common skill history."""
    df = pd.DataFrame({
        "order_id": [1, 2, 3],
        "user_id": ["u1", "u1", "u1"],
        "problem_id": ["p1", "p2", "p3"],
        "correct": [1, 0, 1],
        "skill_id": [None, None, "1"],
        "skill_name": [None, None, "SkillA"],
        "attempt_count": [1, 1, 1],
        "hint_count": [0, 0, 0],
    })
    cleaned, _ = clean_and_validate_data(df)
    norm, _, _ = normalize_skills(cleaned)
    res = engineer_historical_features(norm)

    # Untagged rows must have NaN for skill-specific historical features
    assert np.isnan(res.loc[0, "user_skill_prev_count"])
    assert np.isnan(res.loc[0, "user_skill_prev_accuracy"])
    assert np.isnan(res.loc[1, "user_skill_prev_count"])
    assert np.isnan(res.loc[1, "user_skill_prev_accuracy"])

    # First row with skill_id '1' must start at count=0, acc=NaN
    assert res.loc[2, "user_skill_prev_count"] == 0
    assert np.isnan(res.loc[2, "user_skill_prev_accuracy"])


def test_combined_skill_ids_normalized_correctly():
    """Test 6: Combined skill IDs are normalized correctly."""
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "user_id": ["u1", "u1", "u1", "u1"],
        "problem_id": ["p1", "p2", "p3", "p4"],
        "correct": [1, 1, 1, 1],
        "skill_id": ["1_13", "2_37_70", "2_37_48_77", "27_321"],
        "skill_name": ["A", "B", "B", "C"],
        "attempt_count": [1, 1, 1, 1],
        "hint_count": [0, 0, 0, 0],
    })
    cleaned, _ = clean_and_validate_data(df)
    norm, _, _ = normalize_skills(cleaned)

    assert list(norm["canonical_skill_id"]) == ["1", "2", "2", "27"]
    assert list(norm["n_skill_components"]) == [2, 3, 4, 2]


def test_ids_85_and_163_remain_distinct():
    """Test 7: IDs 85 and 163 remain distinct despite sharing the name Absolute Value."""
    df = pd.DataFrame({
        "order_id": [1, 2],
        "user_id": ["u1", "u1"],
        "problem_id": ["p1", "p2"],
        "correct": [1, 0],
        "skill_id": ["85", "163"],
        "skill_name": ["Absolute Value", "Absolute Value"],
        "attempt_count": [1, 1],
        "hint_count": [0, 0],
    })
    cleaned, _ = clean_and_validate_data(df)
    norm, _, _ = normalize_skills(cleaned)
    res = engineer_historical_features(norm)

    assert norm.loc[0, "canonical_skill_id"] == "85"
    assert norm.loc[1, "canonical_skill_id"] == "163"

    # Because IDs are distinct, row 1 (skill 163) has 0 previous attempts in skill 163
    assert res.loc[1, "user_skill_prev_count"] == 0
    assert np.isnan(res.loc[1, "user_skill_prev_accuracy"])


def test_probability_target_is_binary():
    """Test 8: Probability target is binary."""
    df = pd.DataFrame({
        "order_id": [1, 2, 3],
        "user_id": ["u1", "u1", "u1"],
        "problem_id": ["p1", "p2", "p3"],
        "correct": ["1", "0", "2"],  # "2" is invalid
        "skill_id": ["1", "1", "1"],
        "skill_name": ["A", "A", "A"],
        "attempt_count": [1, 1, 1],
        "hint_count": [0, 0, 0],
    })
    cleaned, report = clean_and_validate_data(df)

    # Invalid non-binary row should be reported and removed from valid rows
    assert report["invalid_correct_count"] == 1
    assert set(cleaned["correct"].unique()).issubset({0, 1})
    assert len(cleaned) == 2


def test_duplicate_order_ids_handled_explicitly():
    """Test 9: Duplicate order IDs are handled explicitly and reported."""
    df = pd.DataFrame({
        "order_id": [1, 1, 2],
        "user_id": ["u1", "u1", "u2"],
        "problem_id": ["p1", "p1", "p2"],
        "correct": [1, 0, 1],
        "skill_id": ["1", "1", "2"],
        "skill_name": ["A", "A", "B"],
        "attempt_count": [1, 1, 1],
        "hint_count": [0, 0, 0],
    })
    cleaned, report = clean_and_validate_data(df)

    assert report["duplicate_order_ids_count"] == 1
    assert len(cleaned) == 2
    assert cleaned["order_id"].is_unique


def test_raw_data_is_never_modified(tmp_path: Path):
    """Test 10: Raw data file is never modified by the pipeline."""
    raw_file = tmp_path / "raw_assistments.csv"
    raw_content = (
        "order_id,user_id,problem_id,correct,skill_id,skill_name,attempt_count,hint_count,original,ms_first_response\n"
        "1,u1,p1,1,10_1,Math,1,0,1,1000\n"
        "2,u1,p2,0,10_2,Math,2,1,1,2000\n"
    )
    raw_file.write_text(raw_content, encoding="latin-1")
    initial_hash = hash(raw_file.read_bytes())

    out_dir = tmp_path / "processed"
    process_dataset(raw_path=raw_file, output_dir=out_dir)

    final_hash = hash(raw_file.read_bytes())
    assert initial_hash == final_hash
    assert (out_dir / "events.csv").exists()
    assert (out_dir / "skill_catalog.csv").exists()
    assert (out_dir / "problem_skill_map.csv").exists()

