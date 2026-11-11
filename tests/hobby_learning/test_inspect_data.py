"""Unit tests for MOOCCubeX data inspection script."""

import json
from pathlib import Path

from src.hobby_learning.inspect_data import (
    detect_file_format,
    inspect_concept_problem,
    parse_timestamp,
    stream_inspect_user_problem,
)


def test_detect_file_format():
    assert "JSONL" in detect_file_format('{"a": 1}\n{"b": 2}')
    assert "JSON Array" in detect_file_format('[{"a": 1}, {"b": 2}]')
    assert "Empty" in detect_file_format("   ")


def test_parse_timestamp():
    assert parse_timestamp("2020-10-27 10:11:56") is True
    assert parse_timestamp("2020-10-27T10:11:56") is True
    assert parse_timestamp("invalid-date") is False
    assert parse_timestamp(None) is False


def test_inspect_concept_problem(tmp_path: Path):
    sample_file = tmp_path / "concept-problem.txt"
    sample_file.write_text(
        "K_Python_ComputerScience\tPm_101\n"
        "K_Python_ComputerScience\tPm_102\n"
        "K_ML_AI\tPm_103\n",
        encoding="utf-8",
    )

    stats = inspect_concept_problem(sample_file)
    assert stats["total_lines"] == 3
    assert stats["malformed_lines"] == 0
    assert stats["unique_concepts"] == 2
    assert stats["unique_problems"] == 3
    assert stats["delimiter"] == "'\\t'"


def test_stream_inspect_user_problem(tmp_path: Path):
    sample_file = tmp_path / "user-problem.jsonl"
    lines = [
        json.dumps({
            "log_id": "1",
            "problem_id": "Pm_101",
            "user_id": "U_1",
            "is_correct": 1,
            "attempts": 1,
            "score": None,
            "submit_time": "2020-10-27 10:11:56",
        }),
        json.dumps({
            "log_id": "2",
            "problem_id": "Pm_999",
            "user_id": "U_2",
            "is_correct": 0,
            "attempts": 2,
            "score": -1.0,
            "submit_time": "2020-10-27 10:15:00",
        }),
        "malformed_json_line",
    ]
    sample_file.write_text("\n".join(lines), encoding="utf-8")

    concept_problems = {"Pm_101", "Pm_102"}
    stats = stream_inspect_user_problem(sample_file, concept_problems)

    assert stats["total_lines"] == 3
    assert stats["valid_json_records"] == 2
    assert stats["malformed_records"] == 1
    assert stats["matched_records"] == 1
    assert stats["unmatched_records"] == 1
    assert stats["is_correct_usable"] is True
    assert stats["unique_users_count"] == 2
    assert stats["unique_problems_count"] == 2
    assert stats["intersection_problems_count"] == 1

