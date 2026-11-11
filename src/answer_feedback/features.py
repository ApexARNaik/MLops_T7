"""
Feature Engineering for Answer-Sheet Scoring and Feedback Module
Owner: Member 2 (AV)
Project: AI-Powered Student Study Assistant MLOps Project
Path: src/answer_feedback/features.py

Features engineered per student-question pair:
1. tfidf_cos:
   TF-IDF cosine similarity between student answer and reference answer.
   Fitted strictly on reference/permitted training texts to prevent data leakage.
2. coverage:
   Fraction of rubric points matched.
   Normalized with lowercasing and punctuation removal.
   Matches either via exact subphrase or >= 60% token overlap.
3. max_point_hit:
   Binary indicator (1 or 0) whether at least one rubric point was matched.
4. n_words:
   Number of words in student answer.
5. log_words:
   Log-transformed word count: log(1 + n_words).
6. ocr_conf:
   OCR extraction confidence score (defaults to NaN if unattempted/unextracted).
7. question_mean_mark:
   Historical/training mean mark per question. Computed exclusively on training folds.
8. max_mark:
   Maximum possible mark defined for the question (2.0 for all Q21-Q35).

Leakage & Integrity Constraints:
- Never use the student's true ground truth mark as an input feature.
- Never inspect or read Corrected_Pdf.
- Missing answers remain explicitly identifiable (NaN/is_missing flag) and are never replaced with zero marks.
- Vectorizers and statistical targets are calculated in fold-safe isolation.
- Functions are modular and reusable for both training and online inference.
"""

import os
import re
import json
import string
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Default paths
DEFAULT_OCR_CSV = os.path.join("data", "answer_feedback", "interim", "answers_ocr.csv")
DEFAULT_RUBRIC_JSON = os.path.join("data", "answer_feedback", "processed", "rubric.json")
DEFAULT_MARKS_CSV = os.path.join("data", "answer_feedback", "raw", "Teacher_manual_marks_Anonymized.csv")
PROCESSED_MARKS_CSV = os.path.join("data", "answer_feedback", "processed", "marks_long.csv")
OUTPUT_FEATURES_CSV = os.path.join("data", "answer_feedback", "processed", "features.csv")


def normalize_text(text: Optional[str]) -> str:
    """
    Normalizes text by lowercasing, stripping punctuation, and collapsing whitespaces.
    """
    if text is None or pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove punctuation using standard string punctuation
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    # Collapse multiple whitespaces
    return " ".join(text.split())


def compute_word_counts(text: Optional[str]) -> Tuple[int, float]:
    """
    Computes n_words and log_words for a given answer string.
    Returns:
        n_words: integer count of words.
        log_words: natural log of (1 + n_words).
    """
    clean_text = normalize_text(text)
    words = clean_text.split() if clean_text else []
    n_words = len(words)
    log_words = float(np.log1p(n_words))
    return n_words, log_words


def match_rubric_point(student_tokens: set, student_norm: str, point_variants: List[str]) -> bool:
    """
    Checks if a single rubric point matches the student's normalized answer.
    A match occurs if:
    1. The exact variant subphrase appears in the student's answer text, OR
    2. Any variant has >= 60% token overlap with the student's token set.
    """
    if not student_tokens or not student_norm:
        return False

    for variant in point_variants:
        var_norm = normalize_text(variant)
        if not var_norm:
            continue

        # Rule 1: Exact phrase match in normalized text
        if var_norm in student_norm:
            return True

        # Rule 2: At least 60% token overlap with variant tokens
        var_tokens = set(var_norm.split())
        if not var_tokens:
            continue

        overlap = len(student_tokens.intersection(var_tokens))
        overlap_ratio = overlap / len(var_tokens)
        if overlap_ratio >= 0.60:
            return True

    return False


def compute_rubric_coverage(answer_text: Optional[str], rubric_item: Dict[str, Any]) -> Tuple[float, int]:
    """
    Calculates coverage (fraction of rubric points matched) and max_point_hit (1 or 0).
    """
    norm_answer = normalize_text(answer_text)
    student_tokens = set(norm_answer.split()) if norm_answer else set()

    points = rubric_item.get("points", [])
    if not points:
        return 0.0, 0

    hits = 0
    for point_variants in points:
        if match_rubric_point(student_tokens, norm_answer, point_variants):
            hits += 1

    total_points = len(points)
    coverage = float(hits / total_points) if total_points > 0 else 0.0
    max_point_hit = 1 if hits > 0 else 0
    return round(coverage, 4), max_point_hit


class TfidfSimilarityEngine:
    """
    Calculates TF-IDF cosine similarity between student responses and reference answers.
    Ensures safe fitting on reference text (and optionally permitted training text)
    without leaking test fold student answers.
    """
    def __init__(self, rubric: Dict[str, Any], additional_train_texts: Optional[List[str]] = None):
        self.rubric = rubric
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.ref_vectors: Dict[str, Any] = {}
        self._fit_vectorizers(additional_train_texts)

    def _fit_vectorizers(self, additional_train_texts: Optional[List[str]] = None):
        # Fit a dedicated or shared vectorizer using reference answers + rubric keywords
        for qid, q_data in self.rubric.items():
            corpus = [normalize_text(q_data.get("reference", ""))]
            for pt in q_data.get("points", []):
                for syn in pt:
                    norm_syn = normalize_text(syn)
                    if norm_syn:
                        corpus.append(norm_syn)

            if additional_train_texts:
                for text in additional_train_texts:
                    norm_t = normalize_text(text)
                    if norm_t:
                        corpus.append(norm_t)

            vec = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
            vec.fit(corpus)
            self.vectorizers[qid] = vec
            ref_norm = normalize_text(q_data.get("reference", ""))
            self.ref_vectors[qid] = vec.transform([ref_norm])

    def compute_similarity(self, qid: str, answer_text: Optional[str]) -> float:
        """Computes TF-IDF cosine similarity against the question's reference answer."""
        if answer_text is None or pd.isna(answer_text):
            return 0.0
        norm_answer = normalize_text(answer_text)
        if not norm_answer.strip():
            return 0.0
        if qid not in self.vectorizers:
            return 0.0

        vec = self.vectorizers[qid]
        ref_vec = self.ref_vectors[qid]
        ans_vec = vec.transform([norm_answer])
        sim = cosine_similarity(ans_vec, ref_vec)[0][0]
        return round(float(sim), 4)


def compute_question_means(train_df: pd.DataFrame, default_mean: float = 1.0) -> Dict[str, float]:
    """
    Calculates the mean mark per question strictly using training-fold students.
    """
    means = {}
    if "mark" in train_df.columns and "qid" in train_df.columns:
        grouped = train_df.groupby("qid")["mark"].mean()
        for qid, val in grouped.items():
            means[str(qid)] = round(float(val), 4)
    return means


def extract_features_for_pair(
    student_id: str,
    qid: str,
    answer_text: Optional[str],
    ocr_conf: Optional[float],
    rubric: Dict[str, Any],
    tfidf_engine: TfidfSimilarityEngine,
    question_means: Dict[str, float],
    true_mark: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes all specified features for a single student-question pair.
    Suitable for both batch feature generation and real-time inference.
    """
    q_data = rubric.get(qid, {})
    max_mark = float(q_data.get("max_mark", 2.0))

    # 1. tfidf_cos
    tfidf_cos = tfidf_engine.compute_similarity(qid, answer_text)

    # 2. coverage & 3. max_point_hit
    coverage, max_point_hit = compute_rubric_coverage(answer_text, q_data)

    # 4. n_words & 5. log_words
    n_words, log_words = compute_word_counts(answer_text)

    # 6. ocr_conf (None / NaN preserved if missing)
    conf_val = float(ocr_conf) if ocr_conf is not None and not pd.isna(ocr_conf) else np.nan

    # 7. question_mean_mark
    q_mean = question_means.get(qid, 1.0)

    # Missing status
    is_missing = int(answer_text is None or pd.isna(answer_text) or str(answer_text).strip() == "")

    feature_record = {
        "student_id": student_id,
        "qid": qid,
        "tfidf_cos": tfidf_cos,
        "coverage": coverage,
        "max_point_hit": max_point_hit,
        "n_words": n_words,
        "log_words": round(log_words, 4),
        "ocr_conf": round(conf_val, 4) if not np.isnan(conf_val) else np.nan,
        "question_mean_mark": round(q_mean, 4),
        "max_mark": max_mark,
        "is_missing": is_missing
    }

    if true_mark is not None and not pd.isna(true_mark):
        feature_record["mark"] = float(true_mark)

    return feature_record


def generate_feature_dataset(
    ocr_csv: str = DEFAULT_OCR_CSV,
    rubric_json: str = DEFAULT_RUBRIC_JSON,
    marks_csv: str = PROCESSED_MARKS_CSV,
    output_path: str = OUTPUT_FEATURES_CSV
) -> pd.DataFrame:
    """
    Generates the processed feature dataset by combining:
    - Ground-truth marks (student_id, qid, mark)
    - OCR segmented text (answers_ocr.csv)
    - Question Rubric definitions (rubric.json)
    """
    print("=" * 80)
    print("ANSWER FEEDBACK FEATURE ENGINEERING")
    print("=" * 80)
    print(f"OCR File:       {ocr_csv}")
    print(f"Rubric File:    {rubric_json}")
    print(f"Marks File:     {marks_csv}")
    print(f"Output File:    {output_path}")

    # 1. Load rubric
    if not os.path.exists(rubric_json):
        raise FileNotFoundError(f"Rubric file not found: {rubric_json}")
    with open(rubric_json, "r", encoding="utf-8") as f:
        rubric = json.load(f)
    print(f"Loaded rubric for {len(rubric)} questions.")

    # 2. Load Ground-Truth marks
    if not os.path.exists(marks_csv):
        raise FileNotFoundError(f"Marks file not found: {marks_csv}")
    df_marks = pd.read_csv(marks_csv)
    print(f"Loaded ground-truth marks: {len(df_marks)} attempts across {df_marks['student_id'].nunique()} students.")

    # 3. Load OCR interim data
    df_ocr = pd.DataFrame(columns=["student_id", "qid", "answer_text", "ocr_conf"])
    if os.path.exists(ocr_csv):
        df_ocr = pd.read_csv(ocr_csv)
        print(f"Loaded OCR interim data: {len(df_ocr)} answers across {df_ocr['student_id'].nunique()} students.")
    else:
        print("Warning: answers_ocr.csv not found. Features will be initialized with missing answer flags.")

    # Initialize TF-IDF engine strictly on reference & rubric texts
    tfidf_engine = TfidfSimilarityEngine(rubric=rubric)

    # Compute question means across entire dataset (for full dataset generation)
    # Note: When training cross-validation models, call compute_question_means(train_fold) per fold!
    question_means = compute_question_means(df_marks)

    # Merge marks and OCR data
    # Left join ensures all valid ground-truth attempts are retained
    merged = pd.merge(df_marks, df_ocr, on=["student_id", "qid"], how="left")

    records = []
    for _, row in merged.iterrows():
        sid = str(row["student_id"])
        qid = str(row["qid"])
        ans_text = row["answer_text"] if "answer_text" in row and pd.notna(row["answer_text"]) else None
        conf = row["ocr_conf"] if "ocr_conf" in row and pd.notna(row["ocr_conf"]) else np.nan
        t_mark = row["mark"] if "mark" in row else None

        feat = extract_features_for_pair(
            student_id=sid,
            qid=qid,
            answer_text=ans_text,
            ocr_conf=conf,
            rubric=rubric,
            tfidf_engine=tfidf_engine,
            question_means=question_means,
            true_mark=t_mark
        )
        records.append(feat)

    df_features = pd.DataFrame(records)

    # Save features CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_features.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("FEATURE GENERATION SUMMARY")
    print("=" * 80)
    print(f"Total Records Generated: {len(df_features)}")
    print(f"Total Unique Students:   {df_features['student_id'].nunique()}")
    print(f"Total Unique Questions:  {df_features['qid'].nunique()}")
    print(f"Saved Features Dataset:  {output_path}\n")

    print("Missing Values Summary:")
    missing_summary = df_features.isnull().sum()
    for col, m_cnt in missing_summary.items():
        pct = (m_cnt / len(df_features)) * 100
        print(f"  {col:<20}: {m_cnt:4d} missing ({pct:.1f}%)")

    print("\nFeatures Summary Statistics:")
    stat_cols = ["tfidf_cos", "coverage", "max_point_hit", "n_words", "log_words", "ocr_conf", "question_mean_mark"]
    print(df_features[stat_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]])

    return df_features


if __name__ == "__main__":
    generate_feature_dataset()
