"""Data inspection script for Feature A (Hobby-based learning) MOOCCubeX data.

Validates raw dataset files before feature engineering:
- Inspects relations/concept-problem.txt (columns, separator, headers, cardinality)
- Stream-inspects relations/user-problem-sample.jsonl without loading entire file
- Verifies JSON format, available fields, missing values, and malformed records
- Checks is_correct binary usability and submit_time parseability
- Evaluates user_id and problem_id coverage and calculates intersection with concepts
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, Set


def detect_file_format(sample_str: str) -> str:
    """Detect if the content is line-delimited JSON (JSONL), JSON array, or other."""
    stripped = sample_str.strip()
    if not stripped:
        return "Empty / Unknown"
    if stripped.startswith("["):
        return "JSON Array ([...])"
    if stripped.startswith("{"):
        # Check if second line also starts with {
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if len(lines) > 1 and lines[1].startswith("{"):
            return "JSONL (Line-delimited JSON objects)"
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return "Single JSON Object or JSONL"
        except Exception:
            pass
        return "JSONL (Line-delimited JSON objects)"
    return "Non-JSON / Unknown structure"


def inspect_concept_problem(file_path: Path) -> Dict[str, Any]:
    """Inspect concept-problem.txt without modifying the file.

    Returns detected metadata and statistics.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Concept-problem file not found: {file_path}")

    delimiter_counts: Counter[str] = Counter()
    col_counts: Counter[int] = Counter()
    total_lines = 0
    empty_lines = 0
    malformed_lines = 0
    missing_concept_count = 0
    missing_problem_count = 0

    unique_concepts: Set[str] = set()
    unique_problems: Set[str] = set()
    first_few_rows = []

    delimiters_to_test = ["\t", ",", ";", "|", " "]

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        # Sample first 100 lines to detect primary delimiter
        sample_lines = []
        for _ in range(100):
            line = f.readline()
            if not line:
                break
            sample_lines.append(line)

        for line in sample_lines:
            for d in delimiters_to_test:
                parts = line.strip("\r\n").split(d)
                if len(parts) > 1:
                    delimiter_counts[d] += 1

        chosen_delim = delimiter_counts.most_common(1)[0][0] if delimiter_counts else "\t"

        # Rewind to start
        f.seek(0)

        for line_no, line in enumerate(f, start=1):
            raw = line.strip("\r\n")
            if not raw:
                empty_lines += 1
                continue
            total_lines += 1

            parts = [p.strip() for p in raw.split(chosen_delim)]
            col_counts[len(parts)] += 1

            if len(first_few_rows) < 5:
                first_few_rows.append(parts)

            if len(parts) < 2:
                malformed_lines += 1
                continue

            concept, problem = parts[0], parts[1]
            if not concept:
                missing_concept_count += 1
            else:
                unique_concepts.add(concept)

            if not problem:
                missing_problem_count += 1
            else:
                unique_problems.add(problem)

    # Detect header presence: check if first row values look like column names
    has_header = False
    if first_few_rows:
        first_row = first_few_rows[0]
        header_candidates = {"concept", "problem", "concept_id", "problem_id", "concept_name"}
        if any(col.lower() in header_candidates for col in first_row):
            has_header = True

    column_names = (
        first_few_rows[0]
        if has_header
        else ["concept_id (or name)", "problem_id"]
        + [f"extra_col_{i}" for i in range(2, max(col_counts.keys() or [2]))]
    )

    return {
        "file_path": str(file_path),
        "file_size_bytes": file_path.stat().st_size,
        "delimiter": repr(chosen_delim),
        "has_header": has_header,
        "columns": column_names,
        "column_counts_distribution": dict(col_counts),
        "total_lines": total_lines,
        "empty_lines": empty_lines,
        "malformed_lines": malformed_lines,
        "missing_concept_count": missing_concept_count,
        "missing_problem_count": missing_problem_count,
        "unique_concepts": len(unique_concepts),
        "unique_problems": len(unique_problems),
        "unique_problem_ids_set": unique_problems,
        "sample_rows": first_few_rows[:3],
    }


def parse_timestamp(ts: Any) -> bool:
    """Check if timestamp can be parsed into a valid datetime."""
    if ts is None or not isinstance(ts, str):
        return False
    ts = ts.strip()
    if not ts:
        return False
    try:
        datetime.fromisoformat(ts)
        return True
    except (ValueError, TypeError):
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            datetime.strptime(ts, fmt)
            return True
        except ValueError:
            continue
    return False


def stream_inspect_user_problem(
    file_path: Path,
    concept_problem_ids: Set[str],
) -> Dict[str, Any]:
    """Inspect user-problem-sample.jsonl streaming line-by-line without loading entire file into memory.

    Tracks schema fields, missing fields, malformed lines, is_correct validity,
    submit_time parseability, user_id and problem_id coverage, and intersection
    with concept_problem_ids.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"User problem file not found: {file_path}")

    file_size = file_path.stat().st_size
    field_counts: Counter[str] = Counter()
    field_null_counts: Counter[str] = Counter()
    is_correct_distribution: Counter[Any] = Counter()

    total_lines = 0
    empty_lines = 0
    malformed_records = 0
    valid_json_records = 0

    missing_user_ids = 0
    missing_problem_ids = 0
    missing_is_correct = 0
    missing_submit_time = 0

    submit_time_parsed = 0
    submit_time_failed = 0

    unique_users: Set[str] = set()
    unique_problems: Set[str] = set()

    matched_records = 0
    unmatched_records = 0

    detected_format = "Unknown"
    sample_raw_chunk = ""

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        sample_raw_chunk = f.read(2048)
        detected_format = detect_file_format(sample_raw_chunk)
        f.seek(0)

        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                empty_lines += 1
                continue
            total_lines += 1

            try:
                record = json.loads(raw)
            except Exception:
                malformed_records += 1
                continue

            if not isinstance(record, dict):
                malformed_records += 1
                continue

            valid_json_records += 1

            for k in record.keys():
                field_counts[k] += 1
                if record[k] is None:
                    field_null_counts[k] += 1

            # Check user_id
            u_id = record.get("user_id")
            if u_id is None or u_id == "":
                missing_user_ids += 1
            else:
                unique_users.add(str(u_id))

            # Check problem_id
            p_id = record.get("problem_id")
            if p_id is None or p_id == "":
                missing_problem_ids += 1
                p_str = None
            else:
                p_str = str(p_id)
                unique_problems.add(p_str)

            # Check intersection with concept_problem_ids
            if p_str is not None:
                if p_str in concept_problem_ids:
                    matched_records += 1
                else:
                    unmatched_records += 1

            # Check is_correct
            if "is_correct" not in record or record["is_correct"] is None:
                missing_is_correct += 1
            else:
                val = record["is_correct"]
                is_correct_distribution[val] += 1

            # Check submit_time
            if "submit_time" not in record or record["submit_time"] is None or record["submit_time"] == "":
                missing_submit_time += 1
            else:
                if parse_timestamp(record["submit_time"]):
                    submit_time_parsed += 1
                else:
                    submit_time_failed += 1

    # Evaluate is_correct usability
    total_is_correct_vals = sum(is_correct_distribution.values())
    binary_keys = {0, 1, 0.0, 1.0, "0", "1", True, False}
    valid_binary_count = sum(cnt for val, cnt in is_correct_distribution.items() if val in binary_keys)
    is_correct_usable = total_is_correct_vals > 0 and (valid_binary_count / total_is_correct_vals) >= 0.99

    # Intersection analysis
    intersection_problems = unique_problems.intersection(concept_problem_ids)

    return {
        "file_path": str(file_path),
        "file_size_bytes": file_size,
        "detected_format": detected_format,
        "total_lines": total_lines,
        "empty_lines": empty_lines,
        "valid_json_records": valid_json_records,
        "malformed_records": malformed_records,
        "field_counts": dict(field_counts),
        "field_null_counts": dict(field_null_counts),
        "unique_users_count": len(unique_users),
        "missing_user_ids": missing_user_ids,
        "unique_problems_count": len(unique_problems),
        "missing_problem_ids": missing_problem_ids,
        "is_correct_usable": is_correct_usable,
        "is_correct_distribution": dict(is_correct_distribution),
        "missing_is_correct": missing_is_correct,
        "submit_time_parsed": submit_time_parsed,
        "submit_time_failed": submit_time_failed,
        "missing_submit_time": missing_submit_time,
        "matched_records": matched_records,
        "unmatched_records": unmatched_records,
        "intersection_problems_count": len(intersection_problems),
    }


def format_report(
    cp_stats: Dict[str, Any],
    up_stats: Dict[str, Any],
) -> str:
    """Format inspection findings into a readable terminal report."""
    border = "=" * 80
    sub_border = "-" * 80

    lines = [
        "",
        border,
        "             MOOCCUBEX DATASET INSPECTION & VALIDATION REPORT",
        border,
        "",
        "1. FILE: concept-problem.txt",
        sub_border,
        f"  Path                  : {cp_stats['file_path']}",
        f"  File Size             : {cp_stats['file_size_bytes']:,} bytes ({cp_stats['file_size_bytes'] / (1024 * 1024):.2f} MB)",
        f"  Detected Delimiter    : {cp_stats['delimiter']} (tab)",
        f"  Has Header Row        : {cp_stats['has_header']}",
        f"  Reported Columns      : {', '.join(cp_stats['columns'])}",
        f"  Column Distribution   : {cp_stats['column_counts_distribution']}",
        f"  Total Data Lines      : {cp_stats['total_lines']:,}",
        f"  Malformed Lines       : {cp_stats['malformed_lines']:,}",
        f"  Missing Concept IDs   : {cp_stats['missing_concept_count']:,}",
        f"  Missing Problem IDs   : {cp_stats['missing_problem_count']:,}",
        f"  Unique Concepts       : {cp_stats['unique_concepts']:,}",
        f"  Unique Problem IDs    : {cp_stats['unique_problems']:,}",
        "  Sample Rows (First 3) :",
    ]
    for row in cp_stats["sample_rows"]:
        lines.append(f"    - {row}")

    lines.extend(
        [
            "",
            "2. FILE: user-problem-sample.jsonl",
            sub_border,
            f"  Path                  : {up_stats['file_path']}",
            f"  File Size             : {up_stats['file_size_bytes']:,} bytes ({up_stats['file_size_bytes'] / (1024 * 1024):.2f} MB)",
            f"  Detected JSON Format  : {up_stats['detected_format']}",
            f"  Total Lines Read      : {up_stats['total_lines']:,}",
            f"  Valid JSON Records    : {up_stats['valid_json_records']:,}",
            f"  Malformed Records     : {up_stats['malformed_records']:,}",
            "",
            "  Available Fields & Null Counts:",
        ]
    )

    for field, count in up_stats["field_counts"].items():
        nulls = up_stats["field_null_counts"].get(field, 0)
        pct = (nulls / count * 100) if count > 0 else 0
        lines.append(f"    * {field:<14} : present in {count:,} records | null in {nulls:,} ({pct:.1f}%)")

    lines.extend(
        [
            "",
            "3. FIELD USABILITY CHECKS",
            sub_border,
            f"  User ID Availability : {up_stats['unique_users_count']:,} unique users, {up_stats['missing_user_ids']:,} missing",
            f"  Problem ID Coverage  : {up_stats['unique_problems_count']:,} unique problems, {up_stats['missing_problem_ids']:,} missing",
            f"  is_correct Binary    : Usable = {up_stats['is_correct_usable']}",
            f"    Distribution       : {up_stats['is_correct_distribution']}",
            f"    Missing Values     : {up_stats['missing_is_correct']:,}",
            f"  submit_time Parsing  : Successfully parsed = {up_stats['submit_time_parsed']:,} records",
            f"    Failed Parsing     : {up_stats['submit_time_failed']:,} records",
            f"    Missing Timestamps : {up_stats['missing_submit_time']:,} records",
            "",
            "4. PROBLEM ID INTERSECTION & OVERLAP",
            sub_border,
        ]
    )

    total_valid = up_stats["valid_json_records"]
    matched = up_stats["matched_records"]
    unmatched = up_stats["unmatched_records"]
    matched_pct = (matched / total_valid * 100) if total_valid > 0 else 0
    unmatched_pct = (unmatched / total_valid * 100) if total_valid > 0 else 0

    inter_problems = up_stats["intersection_problems_count"]
    total_up_problems = up_stats["unique_problems_count"]
    total_cp_problems = cp_stats["unique_problems"]

    up_coverage_pct = (inter_problems / total_up_problems * 100) if total_up_problems > 0 else 0
    cp_coverage_pct = (inter_problems / total_cp_problems * 100) if total_cp_problems > 0 else 0

    lines.extend(
        [
            f"  Exercise Records Matched   : {matched:,} ({matched_pct:.2f}%)",
            f"  Exercise Records Unmatched : {unmatched:,} ({unmatched_pct:.2f}%)",
            f"  Unique Problems in Sample  : {total_up_problems:,}",
            f"  Unique Problems in Concept : {total_cp_problems:,}",
            f"  Unique Overlapping Problems: {inter_problems:,} ({up_coverage_pct:.2f}% of exercise problems, {cp_coverage_pct:.2f}% of concept problems)",
            "",
            "5. SUMMARY & DATA INTEGRITY VERDICT",
            sub_border,
        ]
    )

    verdict_passed = (
        cp_stats["malformed_lines"] == 0
        and up_stats["is_correct_usable"]
        and up_stats["submit_time_failed"] == 0
        and up_stats["missing_user_ids"] == 0
        and up_stats["missing_problem_ids"] == 0
    )

    if verdict_passed:
        lines.append("  [PASS] Dataset structure is valid and ready for feature engineering.")
    else:
        lines.append("  [WARN] Dataset has issues noted above; verify handling before training.")

    lines.extend(
        [
            f"  [NOTE] Malformed record rate in sample: {up_stats['malformed_records'] / max(up_stats['total_lines'], 1) * 100:.4f}%",
            f"  [NOTE] Overlap: {inter_problems:,} problems have concept annotations in concept-problem.txt.",
            border,
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Inspect and validate MOOCCubeX raw dataset files.")
    parser.add_argument(
        "--concept-problem",
        type=Path,
        default=Path("data/hobby_learning/raw/concept-problem.txt"),
        help="Path to concept-problem.txt mapping file",
    )
    parser.add_argument(
        "--user-problem",
        type=Path,
        default=Path("data/hobby_learning/raw/user-problem-sample.jsonl"),
        help="Path to user-problem-sample.jsonl exercise log file",
    )
    args = parser.parse_args()

    try:
        cp_stats = inspect_concept_problem(args.concept_problem)
        up_stats = stream_inspect_user_problem(
            args.user_problem,
            cp_stats["unique_problem_ids_set"],
        )
        report = format_report(cp_stats, up_stats)
        print(report)
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to inspect datasets: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
