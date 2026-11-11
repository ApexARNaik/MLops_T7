"""
OCR Output Validation Script
Owner: Member 2 (AV) — Validation Role
Project: AI-Powered Student Study Assistant MLOps Project
Path: src/answer_feedback/validate_ocr.py

Validates data/answer_feedback/interim/answers_ocr.csv produced by BA's OCR pipeline.
Does NOT modify BA's OCR implementation or fabricate missing answers.

Tasks performed:
  1. Validate expected columns exist.
  2. Count unique students.
  3. Count unique question IDs.
  4. Count valid student-question pairs.
  5. Check duplicate (student_id, qid) pairs.
  6. Check empty / whitespace-only answer_text values.
  7. OCR confidence distribution (percentiles + histogram bins).
  8. Coverage against expected pairs (n_pdfs × 15 questions).
  9. Identify incorrectly segmented question IDs (outside Q21–Q35).
 10. Print 5 sample extracted answers (student IDs anonymised as S-N).
 11. Save validation report to docs/ocr_validation.md.
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OCR_CSV = os.path.join(REPO_ROOT, "data", "answer_feedback", "interim", "answers_ocr.csv")
PDF_DIR = os.path.join(REPO_ROOT, "data", "answer_feedback", "raw", "Student_Pdf")
MARKS_CSV = os.path.join(REPO_ROOT, "data", "answer_feedback", "raw",
                         "Teacher_manual_marks_Anonymized.csv")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
REPORT_PATH = os.path.join(DOCS_DIR, "ocr_validation.md")

EXPECTED_COLUMNS = ["student_id", "qid", "answer_text", "ocr_conf", "source"]
VALID_QIDS = {f"Q{i}" for i in range(21, 36)}  # Q21 … Q35
COVERAGE_FALLBACK_THRESHOLD = 0.60              # recommend manual transcription below this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _anon_student(sid: str, mapping: dict) -> str:
    """Return a stable anonymised label (S-1, S-2, …) for a student_id."""
    if sid not in mapping:
        mapping[sid] = f"S-{len(mapping) + 1}"
    return mapping[sid]


def _conf_histogram(series: pd.Series, bins=5) -> str:
    """Return a small ASCII histogram for a confidence series."""
    counts, edges = np.histogram(series.dropna(), bins=bins, range=(0.0, 1.0))
    lines = []
    for i, (lo, hi, cnt) in enumerate(zip(edges, edges[1:], counts)):
        bar = "█" * min(cnt, 40)
        lines.append(f"  [{lo:.1f}–{hi:.1f}]  {bar} {cnt}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def validate_ocr_csv(csv_path: str = OCR_CSV) -> dict:
    results = {}
    anon_map: dict = {}

    # ------------------------------------------------------------------
    # 0. File existence
    # ------------------------------------------------------------------
    if not os.path.exists(csv_path):
        results["file_exists"] = False
        results["fatal"] = f"File not found: {csv_path}"
        return results
    results["file_exists"] = True
    results["csv_path"] = csv_path
    results["file_size_kb"] = round(os.path.getsize(csv_path) / 1024, 1)

    df = pd.read_csv(csv_path)
    results["raw_rows"] = len(df)
    results["raw_cols"] = list(df.columns)

    # ------------------------------------------------------------------
    # 1. Column validation
    # ------------------------------------------------------------------
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    results["missing_columns"] = missing_cols
    results["extra_columns"] = extra_cols
    results["columns_ok"] = len(missing_cols) == 0

    if not results["columns_ok"]:
        results["fatal"] = f"Missing columns: {missing_cols}"
        return results

    # ------------------------------------------------------------------
    # 2. Unique students
    # ------------------------------------------------------------------
    results["unique_students"] = int(df["student_id"].nunique())
    results["student_ids_found"] = sorted(df["student_id"].unique().tolist())

    # Determine expected student count from PDF dir
    if os.path.exists(PDF_DIR):
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
        results["expected_students"] = len(pdf_files)
        results["pdf_files_found"] = len(pdf_files)
    else:
        results["expected_students"] = 50  # known from inspection
        results["pdf_files_found"] = "dir_missing"

    # ------------------------------------------------------------------
    # 3. Unique question IDs
    # ------------------------------------------------------------------
    results["unique_qids"] = sorted(df["qid"].unique().tolist())
    results["n_unique_qids"] = int(df["qid"].nunique())

    # ------------------------------------------------------------------
    # 4. Valid student-question pairs
    # ------------------------------------------------------------------
    valid_mask = df["qid"].isin(VALID_QIDS)
    df_valid = df[valid_mask].copy()
    results["valid_pairs"] = int(valid_mask.sum())
    results["invalid_qid_rows"] = int((~valid_mask).sum())

    # ------------------------------------------------------------------
    # 5. Duplicate (student_id, qid) pairs
    # ------------------------------------------------------------------
    dup_mask = df_valid.duplicated(subset=["student_id", "qid"], keep=False)
    dup_df = df_valid[dup_mask]
    results["duplicate_pairs"] = int(dup_mask.sum())
    results["duplicate_examples"] = []
    if not dup_df.empty:
        for (sid, qid), grp in dup_df.groupby(["student_id", "qid"]):
            results["duplicate_examples"].append({
                "student_anon": _anon_student(sid, anon_map),
                "qid": qid,
                "count": len(grp)
            })

    # ------------------------------------------------------------------
    # 6. Empty / whitespace-only answer_text
    # ------------------------------------------------------------------
    empty_mask = df_valid["answer_text"].isna() | (df_valid["answer_text"].str.strip() == "")
    results["empty_answer_rows"] = int(empty_mask.sum())
    results["non_empty_valid_pairs"] = int((~empty_mask).sum())

    # ------------------------------------------------------------------
    # 7. OCR confidence distribution
    # ------------------------------------------------------------------
    conf_series = df_valid["ocr_conf"].dropna()
    if len(conf_series) > 0:
        results["conf_stats"] = {
            "count": int(len(conf_series)),
            "mean":  round(float(conf_series.mean()), 4),
            "std":   round(float(conf_series.std()), 4),
            "min":   round(float(conf_series.min()), 4),
            "p25":   round(float(conf_series.quantile(0.25)), 4),
            "p50":   round(float(conf_series.quantile(0.50)), 4),
            "p75":   round(float(conf_series.quantile(0.75)), 4),
            "max":   round(float(conf_series.max()), 4),
        }
        results["conf_low_pct"] = round(float((conf_series < 0.40).mean() * 100), 1)
        results["conf_mid_pct"] = round(float(((conf_series >= 0.40) & (conf_series < 0.70)).mean() * 100), 1)
        results["conf_high_pct"] = round(float((conf_series >= 0.70).mean() * 100), 1)
        results["conf_histogram"] = _conf_histogram(conf_series)
    else:
        results["conf_stats"] = {}
        results["conf_low_pct"] = None

    # ------------------------------------------------------------------
    # 8. Coverage calculation
    # ------------------------------------------------------------------
    n_students_in_csv = results["unique_students"]
    expected_students = results["expected_students"]

    # Theoretical max: every student answers every short-answer question
    theoretical_max = expected_students * 15
    results["theoretical_max_pairs"] = theoretical_max

    # Actual attempted (from marks CSV, known from inspection: 568)
    actual_attempted = None
    if os.path.exists(MARKS_CSV):
        try:
            marks_df = pd.read_csv(MARKS_CSV)
            sa_cols = [str(i) for i in range(21, 36)]
            present_cols = [c for c in sa_cols if c in marks_df.columns]
            actual_attempted = int(marks_df[present_cols].notnull().sum().sum())
        except Exception as e:
            actual_attempted = None
            results["marks_csv_error"] = str(e)
    results["actual_attempted_gt"] = actual_attempted  # ground-truth from teacher CSV

    n_valid_non_empty = results["non_empty_valid_pairs"]
    results["coverage_vs_theoretical"] = round(n_valid_non_empty / theoretical_max, 4) if theoretical_max > 0 else 0.0
    results["coverage_vs_theoretical_pct"] = round(results["coverage_vs_theoretical"] * 100, 1)

    if actual_attempted:
        results["coverage_vs_attempted"] = round(n_valid_non_empty / actual_attempted, 4)
        results["coverage_vs_attempted_pct"] = round(results["coverage_vs_attempted"] * 100, 1)
    else:
        results["coverage_vs_attempted"] = None
        results["coverage_vs_attempted_pct"] = None

    # Students not covered at all
    all_pdf_students: list = []
    if os.path.exists(PDF_DIR):
        all_pdf_students = [os.path.splitext(f)[0] for f in os.listdir(PDF_DIR)
                            if f.lower().endswith(".pdf")]
    covered_students = set(df["student_id"].unique())
    missing_students = [s for s in sorted(all_pdf_students) if s not in covered_students]
    results["students_with_no_ocr_data"] = missing_students
    results["n_missing_students"] = len(missing_students)

    # Per-question coverage
    per_q = {}
    for qid in sorted(VALID_QIDS, key=lambda q: int(q[1:])):
        cnt = int((df_valid[df_valid["qid"] == qid]["answer_text"].str.strip().str.len() > 0).sum())
        per_q[qid] = cnt
    results["per_question_coverage"] = per_q

    # ------------------------------------------------------------------
    # 9. Incorrectly segmented question IDs
    # ------------------------------------------------------------------
    invalid_qids = df[~df["qid"].isin(VALID_QIDS)]["qid"].value_counts().to_dict()
    results["invalid_qids_found"] = invalid_qids

    # ------------------------------------------------------------------
    # 10. Five sample answers (anonymised)
    # ------------------------------------------------------------------
    samples = []
    sample_df = df_valid[~empty_mask].copy()
    # Pick a spread across different questions if possible
    if len(sample_df) >= 5:
        sampled = sample_df.groupby("qid").first().reset_index().head(5)
        if len(sampled) < 5:
            sampled = sample_df.sample(n=5, random_state=42)
    else:
        sampled = sample_df

    for _, row in sampled.iterrows():
        samples.append({
            "student_anon": _anon_student(row["student_id"], anon_map),
            "qid": row["qid"],
            "answer_preview": str(row["answer_text"])[:120].replace("\n", " "),
            "ocr_conf": round(float(row["ocr_conf"]), 3)
        })
    results["sample_answers"] = samples

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    cov = results["coverage_vs_theoretical"]
    low_conf_pct = results.get("conf_low_pct", 100)
    recommend_fallback = (cov < COVERAGE_FALLBACK_THRESHOLD) or (low_conf_pct is not None and low_conf_pct > 50)
    results["recommend_manual_fallback"] = recommend_fallback

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_markdown_report(r: dict) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    lines += [
        "# OCR Output Validation Report",
        "",
        f"> **Generated:** {now}  ",
        "> **Validator:** Member 2 (AV)  ",
        "> **Source file:** BA's OCR pipeline → `data/answer_feedback/interim/answers_ocr.csv`  ",
        "> **Policy:** No modification of BA's OCR code; no fabrication of missing answers.",
        "",
    ]

    # Fatal path
    if not r.get("file_exists", False):
        lines += [
            "> [!CAUTION]",
            f"> **File not found:** `{r.get('csv_path', OCR_CSV)}`  ",
            "> The OCR pipeline has not yet produced output. Run BA's pipeline first:",
            "> ```",
            "> python -m src.answer_feedback.ocr_local",
            "> ```",
        ]
        return "\n".join(lines)

    if "fatal" in r:
        lines += [
            "> [!CAUTION]",
            f"> **Fatal validation error:** {r['fatal']}",
        ]
        return "\n".join(lines)

    # ---- File overview ----
    lines += [
        "## 1. File Overview",
        "",
        f"| Property | Value |",
        f"|---|---|",
        f"| Path | `{r['csv_path']}` |",
        f"| File size | {r['file_size_kb']} KB |",
        f"| Total rows (raw) | {r['raw_rows']} |",
        f"| Columns found | `{', '.join(r['raw_cols'])}` |",
        "",
    ]

    # ---- Column check ----
    lines += ["## 2. Column Validation", ""]
    if r["columns_ok"]:
        lines += ["> [!NOTE]", "> ✅ All 5 expected columns present.", ""]
    else:
        lines += [
            "> [!CAUTION]",
            f"> ❌ Missing columns: `{r['missing_columns']}`",
            "",
        ]
    if r["extra_columns"]:
        lines += [f"**Extra columns (non-breaking):** `{r['extra_columns']}`", ""]

    # ---- Students ----
    lines += [
        "## 3. Student Coverage",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Students in CSV | {r['unique_students']} |",
        f"| Expected (PDFs on disk) | {r['expected_students']} |",
        f"| Students with zero OCR data | {r['n_missing_students']} |",
        "",
    ]
    if r["students_with_no_ocr_data"]:
        missing_list = ", ".join(f"`{s}`" for s in r["students_with_no_ocr_data"])
        lines += [f"**Students not covered:** {missing_list}", ""]

    # ---- Question IDs ----
    lines += [
        "## 4. Question ID Coverage",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Unique QIDs in CSV | {r['n_unique_qids']} |",
        f"| Valid QIDs (Q21–Q35) | {len([q for q in r['unique_qids'] if q in VALID_QIDS])} |",
        f"| Invalid / unexpected QIDs | {len(r['invalid_qids_found'])} |",
        "",
    ]
    lines += ["**QIDs found:** `" + "`, `".join(r["unique_qids"]) + "`", ""]

    # ---- Pair counts ----
    lines += [
        "## 5. Student–Question Pair Counts",
        "",
        f"| Metric | Count |",
        f"|---|---|",
        f"| Total rows | {r['raw_rows']} |",
        f"| Valid pairs (QID in Q21–Q35) | {r['valid_pairs']} |",
        f"| Invalid QID rows | {r['invalid_qid_rows']} |",
        f"| Non-empty valid pairs | {r['non_empty_valid_pairs']} |",
        f"| Empty / blank answer_text | {r['empty_answer_rows']} |",
        "",
    ]

    # ---- Duplicates ----
    lines += ["## 6. Duplicate Pairs", ""]
    if r["duplicate_pairs"] == 0:
        lines += ["> [!NOTE]", "> ✅ No duplicate (student_id, qid) pairs found.", ""]
    else:
        lines += [
            "> [!WARNING]",
            f"> ⚠️ **{r['duplicate_pairs']} rows** involved in duplicate (student_id, qid) pairs.",
            "",
            "| Student (anon) | QID | Occurrences |",
            "|---|---|---|",
        ]
        for ex in r["duplicate_examples"][:10]:
            lines.append(f"| {ex['student_anon']} | {ex['qid']} | {ex['count']} |")
        lines.append("")

    # ---- Empty answers ----
    lines += ["## 7. Empty Answer Text", ""]
    if r["empty_answer_rows"] == 0:
        lines += ["> [!NOTE]", "> ✅ No empty answer_text values in valid pairs.", ""]
    else:
        lines += [
            "> [!WARNING]",
            f"> ⚠️ **{r['empty_answer_rows']}** valid pairs have empty answer_text.",
            "> These must NOT be treated as zero-mark answers — they indicate OCR failure.",
            "",
        ]

    # ---- Confidence ----
    lines += ["## 8. OCR Confidence Distribution", ""]
    cs = r.get("conf_stats", {})
    if cs:
        lines += [
            f"| Statistic | Value |",
            f"|---|---|",
            f"| Count | {cs['count']} |",
            f"| Mean | {cs['mean']} |",
            f"| Std Dev | {cs['std']} |",
            f"| Min | {cs['min']} |",
            f"| 25th pct | {cs['p25']} |",
            f"| Median | {cs['p50']} |",
            f"| 75th pct | {cs['p75']} |",
            f"| Max | {cs['max']} |",
            "",
            f"**Confidence bands:**",
            f"- 🔴 Low  (< 0.40): {r['conf_low_pct']}%",
            f"- 🟡 Mid  (0.40–0.70): {r['conf_mid_pct']}%",
            f"- 🟢 High (≥ 0.70): {r['conf_high_pct']}%",
            "",
            "**Histogram (0.0 → 1.0):**",
            "```",
        ]
        lines.append(r.get("conf_histogram", ""))
        lines += ["```", ""]
    else:
        lines += ["> [!WARNING]", "> No confidence data available (empty DataFrame).", ""]

    # ---- Coverage ----
    lines += [
        "## 9. Coverage Analysis",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Theoretical max pairs ({r['expected_students']} students × 15 Qs) | {r['theoretical_max_pairs']} |",
        f"| Ground-truth attempted (from teacher CSV) | {r.get('actual_attempted_gt', 'N/A')} |",
        f"| Non-empty valid pairs extracted | {r['non_empty_valid_pairs']} |",
        f"| **Coverage vs theoretical max** | **{r['coverage_vs_theoretical_pct']}%** |",
    ]
    if r.get("coverage_vs_attempted_pct") is not None:
        lines.append(f"| **Coverage vs actually attempted** | **{r['coverage_vs_attempted_pct']}%** |")
    lines.append("")

    # Per-question table
    lines += [
        "**Per-question segmented answer count (non-empty):**",
        "",
        "| QID | Students Segmented | Coverage (of 50) |",
        "|---|---|---|",
    ]
    for qid, cnt in r["per_question_coverage"].items():
        pct = round(cnt / r["expected_students"] * 100, 0)
        flag = " ⚠️" if pct < 40 else ""
        lines.append(f"| {qid} | {cnt} | {pct:.0f}%{flag} |")
    lines.append("")

    # ---- Invalid QIDs ----
    lines += ["## 10. Incorrectly Segmented Question IDs", ""]
    if not r["invalid_qids_found"]:
        lines += ["> [!NOTE]", "> ✅ No incorrectly segmented QIDs found.", ""]
    else:
        lines += [
            "> [!WARNING]",
            "> ⚠️ The following QIDs are outside Q21–Q35 and should not appear in the output:",
            "",
            "| QID | Row Count |",
            "|---|---|",
        ]
        for qid, cnt in sorted(r["invalid_qids_found"].items()):
            lines.append(f"| `{qid}` | {cnt} |")
        lines.append("")

    # ---- Samples ----
    lines += ["## 11. Sample Extracted Answers (Anonymised)", ""]
    if r["sample_answers"]:
        lines += [
            "| Student | QID | Answer Preview (120 chars) | Conf |",
            "|---|---|---|---|",
        ]
        for s in r["sample_answers"]:
            preview = s["answer_preview"].replace("|", "\\|")
            lines.append(f"| {s['student_anon']} | {s['qid']} | {preview} | {s['ocr_conf']} |")
        lines.append("")
    else:
        lines += ["> No extractable sample answers found.", ""]

    # ---- Recommendation ----
    lines += ["## 12. Recommendation", ""]
    cov_pct = r["coverage_vs_theoretical_pct"]
    low_conf = r.get("conf_low_pct")

    if r["recommend_manual_fallback"]:
        lines += [
            "> [!CAUTION]",
            "> ### ❌ Manual Transcription Fallback Recommended",
            ">",
            f"> **Coverage is {cov_pct}%** (threshold: 60%).",
        ]
        if low_conf is not None:
            lines.append(f"> **{low_conf}% of answers have confidence < 0.40** — text is likely unreadable.")
        lines += [
            ">",
            "> **Recommended actions:**",
            "> 1. Do not use OCR output as the sole source of answer text.",
            "> 2. Flag all rows with `ocr_conf < 0.60` for human review.",
            "> 3. Consider manual transcription for a representative subset to validate rubric scoring.",
            "> 4. BA should investigate page-layout assumptions: question markers may be absent or unreadable in some PDFs.",
            "> 5. As a fallback, use **teacher marks CSV** (already available) as the ground-truth label source for model training.",
            "",
        ]
    else:
        lines += [
            "> [!NOTE]",
            f"> ### ✅ OCR Coverage Acceptable ({cov_pct}%)",
            ">",
            "> Coverage exceeds 60% threshold. Proceed with rubric-matching scoring pipeline.",
            "> Still recommended: human spot-check on low-confidence rows (`ocr_conf < 0.60`).",
            "",
        ]

    # ---- Limitations ----
    lines += [
        "## 13. Known OCR Limitations",
        "",
        "- Handwritten cursive text is inherently difficult for EasyOCR (trained mostly on printed text).",
        "- Question number markers (e.g. `21.`, `Q21`) may be written differently or absent entirely.",
        "- OCR may misread `21` as `2l`, `'` as `l`, splitting or merging tokens incorrectly.",
        "- Multi-page answer sheets may have question text split across pages.",
        "- Low-confidence segments must **never** be assigned a zero mark; treat as `NaN` / 'review required'.",
        "- These limitations are intrinsic to the dataset, not a defect in BA's pipeline design.",
        "",
        "---",
        f"*Report generated by `src/answer_feedback/validate_ocr.py` at {now}*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("OCR OUTPUT VALIDATION")
    print("=" * 70)

    results = validate_ocr_csv(OCR_CSV)

    # Print summary to stdout
    print(f"File exists:              {results.get('file_exists')}")
    if not results.get("file_exists"):
        print(f"ERROR: {results.get('fatal', 'File not found')}")
        print("Run BA's OCR pipeline first: python -m src.answer_feedback.ocr_local")
        sys.exit(1)

    if "fatal" in results:
        print(f"FATAL: {results['fatal']}")
        sys.exit(1)

    print(f"Raw rows:                 {results['raw_rows']}")
    print(f"Columns OK:               {results['columns_ok']}")
    print(f"Unique students:          {results['unique_students']} / {results['expected_students']} expected")
    print(f"Unique QIDs:              {results['n_unique_qids']}")
    print(f"Valid pairs:              {results['valid_pairs']}")
    print(f"Non-empty valid pairs:    {results['non_empty_valid_pairs']}")
    print(f"Duplicate pairs:          {results['duplicate_pairs']}")
    print(f"Empty answer_text:        {results['empty_answer_rows']}")
    print(f"Invalid QID rows:         {results['invalid_qid_rows']}")
    cs = results.get("conf_stats", {})
    if cs:
        print(f"Avg OCR confidence:       {cs.get('mean', 'N/A')}")
        print(f"Low-conf rows (<0.40):    {results.get('conf_low_pct')}%")
    print(f"Coverage (theoretical):   {results['coverage_vs_theoretical_pct']}%")
    if results.get("coverage_vs_attempted_pct") is not None:
        print(f"Coverage (attempted):     {results['coverage_vs_attempted_pct']}%")
    print(f"Students missing:         {results['n_missing_students']}")
    print(f"Invalid QIDs:             {results['invalid_qids_found']}")
    print(f"Manual fallback needed:   {results['recommend_manual_fallback']}")

    print("\nSample answers (anonymised):")
    for s in results.get("sample_answers", []):
        print(f"  [{s['student_anon']} {s['qid']}] conf={s['ocr_conf']}: {s['answer_preview'][:80]}")

    # Build and save markdown report
    os.makedirs(DOCS_DIR, exist_ok=True)
    report_md = build_markdown_report(results)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nValidation report saved → {REPORT_PATH}")

    return results


if __name__ == "__main__":
    main()

