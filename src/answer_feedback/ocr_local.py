"""
OCR Processing Pipeline for Student Examination Papers
Owner: Member 2 (AV)
Project: AI-Powered Student Study Assistant MLOps Project
Path: src/answer_feedback/ocr_local.py

Pipeline steps:
1. Render raw handwritten student PDFs at 200 DPI using PyMuPDF (fitz).
2. Extract text and token confidences using EasyOCR.
3. Associate extracted answers with the correct student ID (Student_1 to Student_50).
4. Identify and segment short-answer questions Q21-Q35 using regex question markers.
5. Generate answers_ocr.csv: student_id, qid, answer_text, ocr_conf, source="easyocr".
6. Strictly prevent usage of Corrected_Pdf (data leakage rule).
7. Report segmentation coverage, average confidence, and handwriting limitations.
"""

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
from PIL import Image

# Path definitions
RAW_STUDENT_PDF_DIR = os.path.join("data", "answer_feedback", "raw", "Student_Pdf")
INTERIM_DIR = os.path.join("data", "answer_feedback", "interim")
OUTPUT_OCR_CSV = os.path.join(INTERIM_DIR, "answers_ocr.csv")
RAW_MARKS_CSV = os.path.join("data", "answer_feedback", "raw", "Teacher_manual_marks_Anonymized.csv")

TARGET_QIDS = [f"Q{i}" for i in range(21, 36)]


def validate_input_path(path: str):
    """Enforces strict leakage prevention: never touch Corrected_Pdf."""
    normalized = os.path.normpath(path).lower()
    if "corrected_pdf" in normalized or "corrected" in normalized:
        raise ValueError(
            f"DATA LEAKAGE VIOLATION: '{path}' references teacher-corrected PDFs! "
            "Only raw 'Student_Pdf' can be used as OCR model input."
        )


def render_pdf_to_images(pdf_path: str, dpi: int = 200) -> list:
    """
    Renders all pages of a PDF to PIL/NumPy images at the specified DPI.
    Uses PyMuPDF (fitz).
    """
    validate_input_path(pdf_path)
    import pymupdf as fitz  # PyMuPDF (new API)

    doc = fitz.open(pdf_path)
    images = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append((page_idx + 1, np.array(img)))
    doc.close()
    return images


def extract_page_ocr(reader, image_np: np.ndarray) -> list:
    """
    Extracts text blocks with bounding boxes and confidence scores using EasyOCR.
    Returns blocks sorted in natural reading order (top-to-bottom, left-to-right).
    """
    results = reader.readtext(image_np)
    # results format: list of (bbox, text, prob)
    # bbox: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    sorted_results = []
    for bbox, text, prob in results:
        text_clean = str(text).strip()
        if not text_clean:
            continue
        top_y = min(p[1] for p in bbox)
        left_x = min(p[0] for p in bbox)
        sorted_results.append({
            "bbox": bbox,
            "top_y": top_y,
            "left_x": left_x,
            "text": text_clean,
            "conf": float(prob)
        })

    # Sort primarily by vertical position (allowing a small line tolerance), then by horizontal
    sorted_results.sort(key=lambda item: (round(item["top_y"] / 25) * 25, item["left_x"]))
    return sorted_results


def segment_short_answers(extracted_blocks_all_pages: list) -> dict:
    """
    Segments OCR tokens across all pages of a student's answer sheet
    into short-answer questions Q21-Q35 using regex patterns.
    
    Returns a dictionary of:
      qid -> {'answer_text': str, 'ocr_conf': float, 'n_tokens': int}
    """
    # Regex to capture question numbers 21 to 35
    # Matches patterns like: "21.", "21)", "Q21", "Q.21", "21 -", "21:"
    q_marker_regex = re.compile(
        r'^(?:Q(?:uestion)?\.?\s*)?(2[1-9]|3[0-5])[\s\.\)\-:]*(.*)$',
        re.IGNORECASE
    )

    current_qid = None
    question_buffers = {f"Q{i}": {"texts": [], "confs": []} for i in range(21, 36)}

    for block in extracted_blocks_all_pages:
        text = block["text"]
        conf = block["conf"]

        match = q_marker_regex.match(text)
        if match:
            q_num = match.group(1)
            remaining_text = match.group(2).strip()
            current_qid = f"Q{q_num}"

            if remaining_text:
                question_buffers[current_qid]["texts"].append(remaining_text)
                question_buffers[current_qid]["confs"].append(conf)
        else:
            # Check if text is just a number between 21 and 35
            if text in [str(i) for i in range(21, 36)]:
                current_qid = f"Q{text}"
            elif current_qid is not None:
                question_buffers[current_qid]["texts"].append(text)
                question_buffers[current_qid]["confs"].append(conf)

    # Consolidate segmented answers
    segmented = {}
    for qid, buf in question_buffers.items():
        if buf["texts"]:
            full_text = " ".join(buf["texts"]).strip()
            avg_conf = float(np.mean(buf["confs"])) if buf["confs"] else 0.0
            # Only keep if there is non-trivial text
            if len(full_text) > 0:
                segmented[qid] = {
                    "answer_text": full_text,
                    "ocr_conf": round(avg_conf, 4),
                    "n_tokens": len(buf["texts"])
                }
    return segmented


def process_all_student_pdfs(
    pdf_dir: str = RAW_STUDENT_PDF_DIR,
    output_path: str = OUTPUT_OCR_CSV,
    limit: int = None,
    dpi: int = 200
) -> pd.DataFrame:
    """
    Executes the full OCR pipeline across all student PDFs in Student_Pdf/.
    """
    validate_input_path(pdf_dir)
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"Student_Pdf directory not found at {pdf_dir}")

    # Lazy import of easyocr
    import easyocr

    print("=" * 80)
    print("OCR PROCESSING PIPELINE (EASYOCR + PYMUPDF)")
    print("=" * 80)
    print(f"Input Directory:  {pdf_dir}")
    print(f"Output CSV:       {output_path}")
    print(f"Rendering DPI:    {dpi}")
    print(f"Limiting to:      {limit if limit else 'All 50 PDFs'}")

    pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")])
    if limit:
        pdf_files = pdf_files[:limit]

    print(f"Found {len(pdf_files)} PDF files to process.")

    # Initialize EasyOCR Reader (English, CPU default for deterministic execution)
    print("Initializing EasyOCR English model...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    print("EasyOCR reader ready.\n")

    records = []
    total_pages_processed = 0

    for idx, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        # Extract student ID (e.g. 'Student_1' from 'Student_1.pdf')
        student_id = os.path.splitext(pdf_file)[0]
        print(f"[{idx}/{len(pdf_files)}] Processing {student_id} ({pdf_file})...", end="", flush=True)

        pages = render_pdf_to_images(pdf_path, dpi=dpi)
        total_pages_processed += len(pages)

        # Extract OCR tokens across all pages
        all_blocks = []
        for page_num, img_np in pages:
            page_blocks = extract_page_ocr(reader, img_np)
            all_blocks.extend(page_blocks)

        # Segment into short-answer questions Q21-Q35
        segmented = segment_short_answers(all_blocks)
        print(f" -> Found {len(segmented)}/15 questions.")

        for qid, q_data in segmented.items():
            records.append({
                "student_id": student_id,
                "qid": qid,
                "answer_text": q_data["answer_text"],
                "ocr_conf": q_data["ocr_conf"],
                "source": "easyocr"
            })

    # Create interim DataFrame
    df_ocr = pd.DataFrame(records)
    if df_ocr.empty:
        df_ocr = pd.DataFrame(columns=["student_id", "qid", "answer_text", "ocr_conf", "source"])

    # Ensure output directory exists and save CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_ocr.to_csv(output_path, index=False)
    print("\n" + "=" * 80)
    print("OCR PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total PDFs processed:          {len(pdf_files)}")
    print(f"Total PDF pages rendered:      {total_pages_processed}")
    print(f"Total answers segmented:       {len(df_ocr)}")
    print(f"Saved CSV to:                  {output_path}")

    # Coverage and Quality Analysis
    calculate_ocr_metrics(df_ocr, len(pdf_files))

    return df_ocr


def calculate_ocr_metrics(df_ocr: pd.DataFrame, n_students_processed: int):
    """
    Computes segmentation coverage and audits OCR quality and limitations.
    """
    theoretical_max = n_students_processed * 15  # 15 short-answer questions
    segmented_count = len(df_ocr)
    coverage_theoretical = (segmented_count / theoretical_max * 100) if theoretical_max > 0 else 0.0

    print(f"\nSegmentation Coverage (Theoretical Max {theoretical_max}): {coverage_theoretical:.1f}% ({segmented_count}/{theoretical_max})")

    # If ground-truth marks CSV is available, compute coverage relative to actually attempted questions
    if os.path.exists(RAW_MARKS_CSV):
        raw_marks = pd.read_csv(RAW_MARKS_CSV)
        sa_cols = [str(i) for i in range(21, 36)]
        actual_attempts = raw_marks[sa_cols].notnull().sum().sum()
        coverage_attempted = (segmented_count / actual_attempts * 100) if actual_attempts > 0 else 0.0
        print(f"Segmentation Coverage (Ground-Truth Attempted {actual_attempts}): {coverage_attempted:.1f}% ({segmented_count}/{actual_attempts})")

    if not df_ocr.empty:
        print(f"\nAverage OCR Confidence: {df_ocr['ocr_conf'].mean():.4f}")
        print(f"Median OCR Confidence:  {df_ocr['ocr_conf'].median():.4f}")
        print(f"Min OCR Confidence:     {df_ocr['ocr_conf'].min():.4f}")
        print(f"Max OCR Confidence:     {df_ocr['ocr_conf'].max():.4f}")

        # Quality Audit: Low confidence flags (<0.60)
        low_conf = df_ocr[df_ocr["ocr_conf"] < 0.60]
        print(f"Low Confidence Answers (<0.60, require human review): {len(low_conf)} ({len(low_conf)/len(df_ocr)*100:.1f}%)")

        # Questions segmented breakdown
        print("\nSegmented answers per question:")
        q_counts = df_ocr["qid"].value_counts().sort_index()
        for qid, cnt in q_counts.items():
            print(f"  {qid}: {cnt:2d} students segmented (avg conf: {df_ocr[df_ocr['qid']==qid]['ocr_conf'].mean():.2f})")

        # Sample extracted text
        print("\nSample Extracted OCR Text (first 3):")
        for _, row in df_ocr.head(3).iterrows():
            print(f"  [{row['student_id']} - {row['qid']}] (conf: {row['ocr_conf']:.2f}): \"{row['answer_text'][:80]}...\"")

        print("\nCRITICAL ASSESSMENT OF HANDWRITING OCR:")
        print("  - Handwritten text segmentation from student exams is noisy.")
        print("  - Classical OCR struggles with cursive variations, non-standard question numbering, and margin notes.")
        print("  - Missing answers MUST be treated as 'unmatched / review required', NOT zero.")
        print("  - Downstream models must incorporate ocr_conf and fall back to human review when confidence is low.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run local EasyOCR pipeline on student exam PDFs.")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N students (e.g. --limit 5 for fast test)")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF rendering (default 200)")
    parser.add_argument("--output", type=str, default=OUTPUT_OCR_CSV, help="Output CSV path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all_student_pdfs(limit=args.limit, dpi=args.dpi, output_path=args.output)

