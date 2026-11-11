# Answer-Sheet Scoring and Feedback Dataset Inspection Report

**Module:** Feature B (Answer-Sheet Scoring and Feedback)  
**Role:** Member 2 (AV)  
**Project:** AI-Powered Student Study Assistant (MLOps RVCE)  
**Dataset DOI:** [10.17632/sf3kvjwknt.1](https://doi.org/10.17632/sf3kvjwknt.1)  
**Dataset Citation & License:** Dinesh K P, *A Dataset of Digitized Student Examination Papers, Answer Keys, and Manual Evaluations for Automated Grading Research*, Mendeley Data, V1 (2026), **CC BY 4.0**  
**Inspection Date:** 11 Nov 2026  

---

## 1. Executive Summary

This inspection report validates the digitized examination dataset extracted into `data/answer_feedback/raw/`. The dataset comprises university-level Data Science examination records from 50 students, consisting of 20 Multiple Choice Questions (MCQs, 1 mark each) and 15 Short Answer Questions (2 marks each, total 50 marks).

All key raw files (`Question.txt`, `answerkey.txt`, `Teacher_manual_marks_Anonymized.csv`, `Student_Pdf/`, `Corrected_Pdf/`) were successfully inspected via Python (`src/answer_feedback/inspect_dataset.py`).

---

## 2. File Verification & Dataset Inventory

| File / Directory | Status | Format | Description |
|---|---|---|---|
| `Question.txt` | Present | Plain text (138 lines) | Official question sheet (Q1–Q20 MCQs, Q21–Q35 Short Answers) |
| `answerkey.txt` | Present | CSV-like text (36 lines) | Authoritative answer key and reference rubric answers |
| `Teacher_manual_marks_Anonymized.csv` | Present | CSV (50 rows, 40 cols) | Itemized manual evaluations awarded by the teacher |
| `Student_MCQ.csv` | Present | CSV (50 rows, 23 cols) | Student chosen options for MCQs |
| `Student_Pdf/` | Present | Directory (50 files) | Raw digitized student examination papers (Model Input) |
| `Corrected_Pdf/` | Present | Directory (50 files) | Teacher-graded and annotated answer sheets (**NEVER used as model input**) |

---

## 3. CSV Schema, Data Types & Missing Values

### 3.1 Overview
- **Path:** `data/answer_feedback/raw/Teacher_manual_marks_Anonymized.csv`
- **Shape:** `(50, 40)` (50 student rows, 40 columns)
- **Column Categories:**
  - **Metadata & Aggregates (5 columns):** `Student_ID`, `Student_Name`, `MCQ mark`, `2 marks`, `total`
  - **MCQ Responses (20 columns):** `'1'`, `'2'`, `'3'`, ..., `'20'` (Student selected option letters `A`, `B`, `C`, `D`)
  - **Short Answer Marks (15 columns):** `'21'`, `'22'`, ..., `'35'` (Teacher awarded marks)

> [!IMPORTANT]
> **Schema Difference from Initial Assumption:**  
> The actual CSV columns for questions are numeric strings (`'1'`, `'2'`, ..., `'35'`), **not** prefixed with 'Q' (i.e., `'21'` instead of `'Q21'`). The later feature engineering, evaluation, and pipeline steps must adapt to this schema by mapping bare column names `'21'`–`'35'` to canonical IDs `Q21`–`Q35`.

### 3.2 First 5 Rows Preview

```
  Student_ID Student_Name  MCQ mark  2 marks  total  ...   21   22   23   24   25   26   27   28   29   30   31   32   33   34   35
0  Student_1  Anonymous_1        14     21.0   35.0  ...    2  1.0  1.0  2.0  2.0  2.0  1.0  2.0  1.0    2  1.0  NaN  0.0  2.0  2.0
1  Student_2  Anonymous_2        10     13.0   23.0  ...    2  1.5  2.0  1.0  0.0  NaN  NaN  2.0  1.0  NaN  NaN  NaN  2.0  NaN  1.5
2  Student_3  Anonymous_3        11     13.0   24.0  ...    2  2.0  0.0  0.0  1.0  2.0  1.0  1.0  2.0  NaN  1.0    0  NaN  1.0  NaN
3  Student_4  Anonymous_4        15     21.0   36.0  ...    2  2.0  1.0  2.0  2.0  2.0  2.0  2.0  2.0  NaN  2.0    0  NaN  2.0  NaN
4  Student_5  Anonymous_5        11     15.0   26.0  ...    1  1.0  1.0  2.0  2.0  2.0  0.0  2.0  1.0    2  1.0  NaN  NaN  NaN  NaN
```

### 3.3 Missing & Special Coded Values

1. **`NaN` Values (Unattempted Questions):**
   - In questions 21–35, `NaN` represents questions left completely blank/unattempted by students.
   - Unattempted count ranges from 3 students on Q21 to 32 students on Q35.
   - On average, students attempted 11.4 out of 15 short-answer questions (median: 11).
2. **`NC` ("No mark awarded, not corrected"):**
   - Tag explicitly documented by the authors indicating a question was evaluated with 0 marks due to invalidity/omission.
   - Occurrences: 2 total (Student_12 in Col `'21'`, Student_43 in Col `'30'`).
3. **`NM` ("Not mentioned / No mark"):**
   - Similar to NC, evaluated with 0 marks.
   - Occurrences: 1 total (Student_20 in Col `'32'`).

---

## 4. Verification of Short-Answer Questions (Q21–Q35)

Cross-referencing `Question.txt` confirms that **Part II: Short Answer Questions (2 Marks Each)** lists questions 21 through 35:

| QID | CSV Col | Topic in `Question.txt` | Nominal Max Mark |
|---|---|---|---|
| Q21 | `'21'` | Define Supervised Learning | 2 |
| Q22 | `'22'` | Evaluation Case Study (High Train Acc 99%, Low Test Acc 30% - Overfitting) | 2 |
| Q23 | `'23'` | Regression Analysis definition | 2 |
| Q24 | `'24'` | Two major challenges in Machine Learning | 2 |
| Q25 | `'25'` | Two major real-world applications of Data Science | 2 |
| Q26 | `'26'` | Distinguish between Structured and Semi-structured data | 2 |
| Q27 | `'27'` | Outlier detection and handling methods | 2 |
| Q28 | `'28'` | What is Data Pre-processing? | 2 |
| Q29 | `'29'` | Why Data Normalization is required for K-NN | 2 |
| Q30 | `'30'` | Continuous attribute discretization with example | 2 |
| Q31 | `'31'` | Meaning of 'K' in K-Means clustering | 2 |
| Q32 | `'32'` | Three common Hyperparameters in ML algorithms | 2 |
| Q33 | `'33'` | Define Gradient Descent | 2 |
| Q34 | `'34'` | Compare three SVM classifiers based on margins/boundaries | 2 |
| Q35 | `'35'` | Explain how RBF Kernel works in SVM | 2 |

---

## 5. Possible Mark Values & 0.5 Increments

- **Distinct Raw Values in Columns 21–35:** `['0', '0.0', '0.5', '1', '1.0', '1.5', '2', '2.0', '4.0', 'NC', 'NM']`
- **Do 0.5 Increments Exist?:** **YES.**
  - Fractional marks `0.5` and `1.5` occur extensively across 12 of the 15 short-answer questions (e.g., Q22, Q24, Q25, Q26, Q27, Q28, Q29, Q30, Q31, Q33, Q34, Q35).
  - Consequently, the scoring model and rounding post-processor must support step sizes of **0.5** (i.e., allowed mark bins: `0.0, 0.5, 1.0, 1.5, 2.0`).
- **Data Anomaly Identified:**
  - In Column `'34'` for `Student_11`, the mark recorded is `4.0` (exceeding 2.0). The student's recorded `2 marks` total of `28.5` accounts for this `4.0`. In alignment with Rule 8 ("Do not modify or invent dataset values"), this value is preserved as recorded in the raw data, and clipping to `[0, max_mark]` will be handled in evaluation feature layers.

---

## 6. Student Count & Marks Distribution for Q21–Q35

**Total Students:** 50

| Question | Answered | Missing (NaN) | Mark Breakdown (`Value: Count (Pct)`) | Mean Mark* | Std* |
|---|---|---|---|---|---|
| **Q21** | 47 | 3 | `1: 4 (8%), 2: 42 (84%), NC: 1 (2%)` | 1.87 | 0.40 |
| **Q22** | 38 | 12 | `0.0: 2 (4%), 1.0: 21 (42%), 1.5: 1 (2%), 2.0: 14 (28%)` | 1.33 | 0.57 |
| **Q23** | 41 | 9 | `0.0: 4 (8%), 1.0: 20 (40%), 2.0: 17 (34%)` | 1.32 | 0.65 |
| **Q24** | 44 | 6 | `0.0: 4 (8%), 0.5: 1 (2%), 1.0: 14 (28%), 2.0: 25 (50%)` | 1.47 | 0.68 |
| **Q25** | 46 | 4 | `0.0: 5 (10%), 1.0: 4 (8%), 1.5: 1 (2%), 2.0: 36 (72%)` | 1.68 | 0.66 |
| **Q26** | 46 | 4 | `0.0: 4 (8%), 1.0: 6 (12%), 1.5: 2 (4%), 2.0: 34 (68%)` | 1.67 | 0.63 |
| **Q27** | 45 | 5 | `0.0: 3 (6%), 0.5: 1 (2%), 1.0: 11 (22%), 1.5: 1 (2%), 2.0: 29 (58%)` | 1.58 | 0.63 |
| **Q28** | 46 | 4 | `1.0: 10 (20%), 1.5: 1 (2%), 2.0: 35 (70%)` | 1.77 | 0.42 |
| **Q29** | 45 | 5 | `0.0: 3 (6%), 0.5: 1 (2%), 1.0: 23 (46%), 2.0: 18 (36%)` | 1.32 | 0.61 |
| **Q30** | 26 | 24 | `0: 5 (10%), 0.5: 1 (2%), 1: 12 (24%), 2: 7 (14%), NC: 1 (2%)` | 1.02 | 0.73 |
| **Q31** | 43 | 7 | `0.0: 2 (4%), 1.0: 16 (32%), 1.5: 1 (2%), 2.0: 24 (48%)` | 1.52 | 0.59 |
| **Q32** | 28 | 22 | `0: 14 (28%), 1: 6 (12%), 2: 7 (14%), NM: 1 (2%)` | 0.71 | 0.85 |
| **Q33** | 21 | 29 | `0.0: 11 (22%), 0.5: 2 (4%), 1.0: 4 (8%), 1.5: 1 (2%), 2.0: 3 (6%)` | 0.60 | 0.75 |
| **Q34** | 34 | 16 | `1.0: 10 (20%), 1.5: 1 (2%), 2.0: 22 (44%), 4.0: 1 (2%)` | 1.75 | 0.61 |
| **Q35** | 18 | 32 | `0.0: 5 (10%), 1.0: 6 (12%), 1.5: 3 (6%), 2.0: 4 (8%)` | 1.03 | 0.76 |

*\*Note: Mean and Std calculated treating NC/NM as 0 on attempted responses.*

---

## 7. Inspection of `Student_Pdf/` Directory

- **File Count:** Exactly **50 PDF files** present (`Student_1.pdf` through `Student_50.pdf`).
- **Total Pages:** 200 scanned handwritten pages across all 50 students.
- **Page Distribution:**
  - 4 pages: 40 PDFs (80%)
  - 6 pages: 4 PDFs (8%)
  - 2 pages: 4 PDFs (8%)
  - 3 pages: 1 PDF (2%)
  - 5 pages: 1 PDF (2%)

---

## 8. Data Leakage Prevention Rule (`Corrected_Pdf`)

> [!CAUTION]
> **STRICT COMPLIANCE MANDATE:**  
> The `Corrected_Pdf/` directory contains 50 teacher-annotated PDFs marked with red pen ticks, corrections, and explicit scores.  
> **Under no circumstances will `Corrected_Pdf` be used as input to OCR, text extraction, feature engineering, or model training.**  
> Using `Corrected_Pdf` would directly leak the evaluation target into the input features, invalidating the model. Only `Student_Pdf/` may be processed by the OCR pipeline.

---

## 9. Next Steps for Implementation (Workstream B)

1. **Rubric Construction:** Parse `answerkey.txt` into `data/answer_feedback/processed/rubric.json` mapping each Q21–Q35 to reference answers and key credit points.
2. **Schema Adapter:** Ensure all data loading routines convert columns `'21'`–`'35'` into standardized question identifiers `Q21`–`Q35`.
3. **Marks Baseline:** Construct the student-grouped `GroupKFold` baseline in `src/answer_feedback/train.py` utilizing question-mean benchmarks.
4. **Integration with OCR Pipeline:** Await segmented OCR transcripts from Member 3 (BA) (`data/answer_feedback/interim/answers_ocr.csv`) and evaluate against the rubric features.

