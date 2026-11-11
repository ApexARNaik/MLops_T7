"""
Data Loading and Processing Module for Answer Feedback
Owner: Member 2 (AV)
Project: AI-Powered Student Study Assistant MLOps Project
Path: src/answer_feedback/data.py

Handles:
- Loading raw marks from Teacher_manual_marks_Anonymized.csv
- Transforming short-answer questions (Q21-Q35) into long format (student_id, qid, mark)
- Preserving actual student IDs and handling NC / NM special values
- Saving processed long-format dataset for training and evaluation
"""

import os
import pandas as pd
import numpy as np

RAW_MARKS_PATH = os.path.join("data", "answer_feedback", "raw", "Teacher_manual_marks_Anonymized.csv")
PROCESSED_LONG_PATH = os.path.join("data", "answer_feedback", "processed", "marks_long.csv")
SHORT_ANSWER_COLS = [str(i) for i in range(21, 36)]


def load_raw_marks(path: str = RAW_MARKS_PATH) -> pd.DataFrame:
    """Loads the raw teacher manual marks CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw marks CSV not found at {path}")
    return pd.read_csv(path)


def build_long_format_marks(
    raw_path: str = RAW_MARKS_PATH,
    output_path: str = PROCESSED_LONG_PATH
) -> pd.DataFrame:
    """
    Transforms the short-answer columns (21-35) from wide to long format:
    Columns: student_id, qid, mark.
    
    Preserves actual student identifiers (Student_ID column).
    Maps special values 'NC' (No mark awarded, not corrected) and 'NM' (Not mentioned) to 0.0.
    Drops NaN values corresponding to unattempted questions.
    """
    df = load_raw_marks(raw_path)
    
    if "Student_ID" not in df.columns:
        raise KeyError(f"'Student_ID' column not found in raw marks CSV. Columns: {list(df.columns)}")
        
    records = []
    for _, row in df.iterrows():
        student_id = str(row["Student_ID"]).strip()
        for col_name in SHORT_ANSWER_COLS:
            if col_name not in df.columns:
                continue
            val = row[col_name]
            
            # Skip unattempted questions (NaN)
            if pd.isna(val):
                continue
                
            # Process special tags and numeric values
            str_val = str(val).strip().upper()
            if str_val in ("NC", "NM"):
                mark = 0.0
            else:
                try:
                    mark = float(val)
                except ValueError:
                    raise ValueError(f"Unexpected non-numeric mark '{val}' for student {student_id}, question {col_name}")
            
            records.append({
                "student_id": student_id,
                "qid": f"Q{col_name}",
                "mark": mark
            })
            
    long_df = pd.DataFrame(records)
    
    # Ensure processed directory exists
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        long_df.to_csv(output_path, index=False)
        print(f"Saved long-format marks to {output_path} (Shape: {long_df.shape})")
        
    return long_df


if __name__ == "__main__":
    long_df = build_long_format_marks()
    print("Sample long-format rows:")
    print(long_df.head(10).to_string(index=False))
    print(f"\nTotal students: {long_df['student_id'].nunique()}")
    print(f"Total samples: {len(long_df)}")
    print(f"Questions covered: {sorted(long_df['qid'].unique())}")

