"""Data engineering and preprocessing pipeline for ASSISTments 2009-2010 dataset.

Feature A: AI-Powered Student Study Assistant MLOps.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "order_id",
    "user_id",
    "problem_id",
    "correct",
    "skill_id",
    "skill_name",
    "attempt_count",
    "hint_count",
    "original",
    "ms_first_response",
]


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load the raw ASSISTments CSV file using latin-1 encoding and dtype=object.

    Validates presence of required columns and ensures raw data is not modified.

    Args:
        file_path: Path to the raw CSV file.

    Returns:
        pd.DataFrame containing the raw dataset with string object types.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If any required column is missing.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset file not found: {path}")

    # Read with latin-1 and dtype=object to avoid UTF-8 string conversion issues
    df = pd.read_csv(path, encoding="latin-1", dtype=object)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    return df


def clean_and_validate_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and validate the raw dataset rows.

    - Validates uniqueness of order_id.
    - Validates correct contains only binary 0/1 values.
    - Converts order_id, correct, attempt_count, and hint_count to numeric types.
    - Keeps user_id and problem_id as stable identifiers (strings).
    - Preserves all valid interactions, including missing skill IDs.
    - Reports missing skill IDs, skill names, and malformed rows.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (cleaned_df, report_dict).
    """
    total_original_rows = len(df)
    report: Dict[str, Any] = {
        "total_original_rows": total_original_rows,
        "malformed_rows_count": 0,
        "duplicate_order_ids_count": 0,
        "invalid_correct_count": 0,
        "missing_skill_id_count": 0,
        "missing_skill_name_count": 0,
    }

    work_df = df.copy()

    # Check for missing values in non-nullable identifier and metric columns
    # Note: skill_id and skill_name are allowed to be missing.
    mandatory_cols = [
        "order_id",
        "user_id",
        "problem_id",
        "correct",
        "attempt_count",
        "hint_count",
    ]
    nan_mask = work_df[mandatory_cols].isna().any(axis=1)

    # Check numeric conversions
    order_id_num = pd.to_numeric(work_df["order_id"], errors="coerce")
    correct_num = pd.to_numeric(work_df["correct"], errors="coerce")
    attempt_num = pd.to_numeric(work_df["attempt_count"], errors="coerce")
    hint_num = pd.to_numeric(work_df["hint_count"], errors="coerce")

    type_error_mask = order_id_num.isna() | correct_num.isna() | attempt_num.isna() | hint_num.isna()

    # Validate binary correct values
    invalid_correct_mask = ~correct_num.isin([0, 1])

    malformed_mask = nan_mask | type_error_mask | invalid_correct_mask
    malformed_count = int(malformed_mask.sum())
    report["malformed_rows_count"] = malformed_count
    report["invalid_correct_count"] = int(invalid_correct_mask.sum())

    if malformed_count > 0:
        print(f"Report: Found {malformed_count} malformed or invalid rows.")

    valid_df = work_df[~malformed_mask].copy()

    # Convert numeric columns
    valid_df["order_id"] = pd.to_numeric(valid_df["order_id"]).astype(np.int64)
    valid_df["correct"] = pd.to_numeric(valid_df["correct"]).astype(np.int64)
    valid_df["attempt_count"] = pd.to_numeric(valid_df["attempt_count"]).astype(np.int64)
    valid_df["hint_count"] = pd.to_numeric(valid_df["hint_count"]).astype(np.int64)

    # Keep user_id and problem_id as stable strings
    valid_df["user_id"] = valid_df["user_id"].astype(str)
    valid_df["problem_id"] = valid_df["problem_id"].astype(str)

    # Check unique order_id
    duplicate_mask = valid_df.duplicated(subset=["order_id"], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    report["duplicate_order_ids_count"] = duplicate_count
    if duplicate_count > 0:
        print(f"Report: Found {duplicate_count} duplicate order_ids. Retaining first occurrence.")
        valid_df = valid_df[~duplicate_mask].copy()

    # Track missing skill information
    report["missing_skill_id_count"] = int(valid_df["skill_id"].isna().sum())
    report["missing_skill_name_count"] = int(valid_df["skill_name"].isna().sum())
    report["valid_rows"] = len(valid_df)

    return valid_df, report


def normalize_skills(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Any]]:
    """Normalize skill IDs and manage canonical skill mappings.

    - canonical_skill_id = first component of skill_id split by '_'
    - Preserves original skill_id column.
    - Creates n_skill_components representing number of components.
    - Builds canonical skill ID to skill name mapping.
    - IDs 85 and 163 both map to 'Absolute Value' and are kept separate.
    - Fills missing skill_name only if canonical ID has a known unambiguous name.
    - Reports any conflicting names.

    Args:
        df: Cleaned interactions DataFrame.

    Returns:
        Tuple of (normalized_df, canonical_skill_map, skill_report).
    """
    norm_df = df.copy()

    has_skill = norm_df["skill_id"].notna()
    norm_df["canonical_skill_id"] = pd.Series(dtype=object)
    norm_df["n_skill_components"] = pd.Series(dtype=float)

    def extract_canonical(val: Any) -> str:
        s = str(val).strip()
        return s.split("_")[0]

    def count_components(val: Any) -> int:
        s = str(val).strip()
        return len(s.split("_"))

    norm_df.loc[has_skill, "canonical_skill_id"] = norm_df.loc[has_skill, "skill_id"].apply(extract_canonical)
    norm_df.loc[has_skill, "n_skill_components"] = (
        norm_df.loc[has_skill, "skill_id"].apply(count_components).astype(np.int64)
    )

    # Check canonical_skill_id to skill_name mapping
    named_mask = has_skill & norm_df["skill_name"].notna()
    skill_groups = norm_df[named_mask].groupby("canonical_skill_id")["skill_name"].unique()

    conflicts: Dict[str, list[str]] = {}
    canonical_map: Dict[str, str] = {}
    for cid, names in skill_groups.items():
        name_list = [str(n) for n in names]
        if len(name_list) > 1:
            conflicts[str(cid)] = name_list
        else:
            canonical_map[str(cid)] = name_list[0]

    skill_report = {
        "conflicts": conflicts,
        "unique_canonical_skills": int(norm_df["canonical_skill_id"].nunique(dropna=True)),
        "unique_skill_names": int(norm_df["skill_name"].nunique(dropna=True)),
    }

    if conflicts:
        print(f"Report: Skill conflicts found for canonical IDs: {conflicts}")

    # Fill missing skill_name if canonical ID has an unambiguous name
    missing_name_mask = has_skill & norm_df["skill_name"].isna()
    inferred_names = norm_df.loc[missing_name_mask, "canonical_skill_id"].map(canonical_map)
    can_fill = inferred_names.notna()
    if can_fill.sum() > 0:
        norm_df.loc[missing_name_mask[can_fill].index, "skill_name"] = inferred_names[can_fill]

    return norm_df, canonical_map, skill_report


def engineer_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create causal historical features without data leakage.

    Sorts by user_id and numeric order_id.
    Calculates cumulative metrics strictly from earlier interactions:
    - user_prev_count
    - user_prev_accuracy
    - user_prev_attempts_mean
    - user_prev_hints_mean

    Skill-specific historical features (grouped by user_id and canonical_skill_id):
    - user_skill_prev_count
    - user_skill_prev_accuracy

    Interactions without a canonical skill contribute to learner-wide history
    but are not grouped together as a shared skill.
    First interactions have count = 0 and averages = NaN.

    Args:
        df: Normalized DataFrame.

    Returns:
        DataFrame sorted by user_id and order_id with added historical features.
    """
    sorted_df = df.sort_values(by=["user_id", "order_id"]).reset_index(drop=True)

    # 1. Learner-wide historical features
    user_grp = sorted_df.groupby("user_id")

    user_prev_count = user_grp.cumcount()
    sorted_df["user_prev_count"] = user_prev_count.astype(np.int64)

    # Cumulative sum minus current row
    cum_correct = user_grp["correct"].cumsum() - sorted_df["correct"]
    cum_attempts = user_grp["attempt_count"].cumsum() - sorted_df["attempt_count"]
    cum_hints = user_grp["hint_count"].cumsum() - sorted_df["hint_count"]

    count_float = user_prev_count.astype(float)
    valid_count = count_float > 0

    sorted_df["user_prev_accuracy"] = np.where(valid_count, cum_correct / count_float, np.nan)
    sorted_df["user_prev_attempts_mean"] = np.where(valid_count, cum_attempts / count_float, np.nan)
    sorted_df["user_prev_hints_mean"] = np.where(valid_count, cum_hints / count_float, np.nan)

    # 2. Skill-specific historical features
    # Untagged interactions must not be grouped together
    has_skill = sorted_df["canonical_skill_id"].notna()

    sorted_df["user_skill_prev_count"] = np.nan
    sorted_df["user_skill_prev_accuracy"] = np.nan

    if has_skill.any():
        skill_subset = sorted_df[has_skill]
        skill_grp = skill_subset.groupby(["user_id", "canonical_skill_id"])

        skill_prev_count = skill_grp.cumcount()
        skill_cum_correct = skill_grp["correct"].cumsum() - skill_subset["correct"]

        skill_count_float = skill_prev_count.astype(float)
        skill_valid = skill_count_float > 0

        skill_accuracy = np.where(skill_valid, skill_cum_correct / skill_count_float, np.nan)

        sorted_df.loc[has_skill, "user_skill_prev_count"] = skill_count_float
        sorted_df.loc[has_skill, "user_skill_prev_accuracy"] = skill_accuracy

    return sorted_df


def build_supporting_tables(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build skill_catalog and problem_skill_map lookup tables.

    Checks consistency of problem-to-skill mappings and reports conflicts.

    Args:
        df: Processed DataFrame.

    Returns:
        Tuple of (skill_catalog_df, problem_skill_map_df, lookup_report).
    """
    has_skill = df["canonical_skill_id"].notna()
    skill_df = df[has_skill]

    # 1. skill_catalog: canonical_skill_id, skill_name
    skill_catalog = (
        skill_df[["canonical_skill_id", "skill_name"]]
        .drop_duplicates(subset=["canonical_skill_id"])
        .sort_values(by="canonical_skill_id")
        .reset_index(drop=True)
    )

    # 2. problem_skill_map: problem_id, canonical_skill_id, skill_name, n_skill_components
    prob_grp = skill_df.groupby("problem_id")["canonical_skill_id"].unique()
    prob_conflicts = {p: list(s) for p, s in prob_grp.items() if len(s) > 1}

    if prob_conflicts:
        print(f"Report: Found {len(prob_conflicts)} problem-to-skill mapping conflicts.")

    problem_skill_map = (
        skill_df[["problem_id", "canonical_skill_id", "skill_name", "n_skill_components"]]
        .drop_duplicates(subset=["problem_id"])
        .sort_values(by="problem_id")
        .reset_index(drop=True)
    )

    lookup_report = {
        "problem_skill_conflicts_count": len(prob_conflicts),
        "catalog_entries": len(skill_catalog),
        "mapped_problems": len(problem_skill_map),
    }

    return skill_catalog, problem_skill_map, lookup_report


def process_dataset(
    raw_path: str | Path = "data/hobby_learning/raw/assistments.csv",
    output_dir: str | Path = "data/hobby_learning/processed/",
) -> Dict[str, Any]:
    """Run full data engineering pipeline and save processed outputs.

    Saves:
        - events.csv
        - skill_catalog.csv
        - problem_skill_map.csv

    Prints comprehensive dataset report.

    Args:
        raw_path: Path to raw assistments CSV.
        output_dir: Destination directory for processed files.

    Returns:
        Summary report dictionary.
    """
    raw_p = Path(raw_path)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Loading Dataset ===")
    df_raw = load_dataset(raw_p)

    print("=== Step 2: Cleaning and Validating Data ===")
    df_clean, clean_report = clean_and_validate_data(df_raw)

    print("=== Step 3: Skill Normalization ===")
    df_norm, canonical_map, skill_report = normalize_skills(df_clean)

    print("=== Step 4: Historical Feature Engineering ===")
    df_events = engineer_historical_features(df_norm)

    print("=== Step 5: Supporting Lookup Tables ===")
    skill_catalog, problem_skill_map, lookup_report = build_supporting_tables(df_events)

    # 6. Main output: events.csv
    # Columns to keep:
    # user_id, order_id, problem_id, original skill_id, canonical_skill_id,
    # skill_name, correct, n_skill_components, historical features
    events_columns = [
        "user_id",
        "order_id",
        "problem_id",
        "skill_id",
        "canonical_skill_id",
        "skill_name",
        "correct",
        "n_skill_components",
        "user_prev_count",
        "user_prev_accuracy",
        "user_prev_attempts_mean",
        "user_prev_hints_mean",
        "user_skill_prev_count",
        "user_skill_prev_accuracy",
    ]

    events_out = df_events[events_columns]

    print("=== Step 6: Saving Processed Files ===")
    events_path = out_p / "events.csv"
    skill_cat_path = out_p / "skill_catalog.csv"
    prob_map_path = out_p / "problem_skill_map.csv"

    events_out.to_csv(events_path, index=False)
    skill_catalog.to_csv(skill_cat_path, index=False)
    problem_skill_map.to_csv(prob_map_path, index=False)

    # Compute report statistics
    total_original_rows = clean_report["total_original_rows"]
    valid_rows = clean_report["valid_rows"]
    n_users = df_events["user_id"].nunique()
    n_problems = df_events["problem_id"].nunique()
    n_canonical_skills = df_events["canonical_skill_id"].nunique(dropna=True)
    n_skill_names = df_events["skill_name"].nunique(dropna=True)
    correct_dist = df_events["correct"].value_counts().to_dict()
    missing_skill_id_count = int(df_events["skill_id"].isna().sum())
    missing_skill_name_count = int(df_events["skill_name"].isna().sum())
    eligible_skill_modelling_count = int(df_events["canonical_skill_id"].notna().sum())
    prob_conflicts_count = lookup_report["problem_skill_conflicts_count"]

    events_size_mb = os.path.getsize(events_path) / (1024 * 1024)
    catalog_size_kb = os.path.getsize(skill_cat_path) / 1024
    prob_map_size_kb = os.path.getsize(prob_map_path) / 1024

    final_report = {
        "total_original_rows": total_original_rows,
        "valid_rows": valid_rows,
        "number_of_users": n_users,
        "number_of_unique_problems": n_problems,
        "number_of_canonical_skill_ids": n_canonical_skills,
        "number_of_skill_names": n_skill_names,
        "correct_incorrect_distribution": correct_dist,
        "missing_skill_id_count": missing_skill_id_count,
        "missing_skill_name_count": missing_skill_name_count,
        "eligible_for_skill_specific_modelling": eligible_skill_modelling_count,
        "problem_to_skill_mapping_conflicts": prob_conflicts_count,
        "output_file_sizes": {
            "events.csv": f"{events_size_mb:.2f} MB",
            "skill_catalog.csv": f"{catalog_size_kb:.2f} KB",
            "problem_skill_map.csv": f"{prob_map_size_kb:.2f} KB",
        },
    }

    print("\n================ DATASET REPORT ================")
    print(f"Total original rows: {total_original_rows}")
    print(f"Valid rows: {valid_rows}")
    print(f"Number of users: {n_users}")
    print(f"Number of unique problems: {n_problems}")
    print(f"Number of canonical skill IDs: {n_canonical_skills}")
    print(f"Number of skill names: {n_skill_names}")
    print(f"Correct/incorrect distribution: {correct_dist}")
    print(f"Missing skill ID count: {missing_skill_id_count}")
    print(f"Missing skill name count: {missing_skill_name_count}")
    print(f"Number of rows eligible for skill-specific modelling: {eligible_skill_modelling_count}")
    print(f"Number of problem-to-skill mapping conflicts: {prob_conflicts_count}")
    print("Output file sizes:")
    for fname, size_str in final_report["output_file_sizes"].items():
        print(f"  - {fname}: {size_str}")
    print("================================================\n")

    return final_report


if __name__ == "__main__":
    process_dataset()
