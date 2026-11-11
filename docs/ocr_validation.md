# OCR Output Validation Report

> **Generated:** 2026-11-11 11:47  
> **Validator:** Member 2 (AV)  
> **Source file:** BA's OCR pipeline → `data/answer_feedback/interim/answers_ocr.csv`  
> **Policy:** No modification of BA's OCR code; no fabrication of missing answers.

## 1. File Overview

| Property | Value |
|---|---|
| Path | `C:\Users\avina_lxw1l2h\Desktop\MLops_T7\data\answer_feedback\interim\answers_ocr.csv` |
| File size | 4.3 KB |
| Total rows (raw) | 15 |
| Columns found | `student_id, qid, answer_text, ocr_conf, source` |

## 2. Column Validation

> [!NOTE]
> ✅ All 5 expected columns present.

## 3. Student Coverage

| Metric | Value |
|---|---|
| Students in CSV | 2 |
| Expected (PDFs on disk) | 50 |
| Students with zero OCR data | 48 |

**Students not covered:** `Student_11`, `Student_12`, `Student_13`, `Student_14`, `Student_15`, `Student_16`, `Student_17`, `Student_18`, `Student_19`, `Student_2`, `Student_20`, `Student_21`, `Student_22`, `Student_23`, `Student_24`, `Student_25`, `Student_26`, `Student_27`, `Student_28`, `Student_29`, `Student_3`, `Student_30`, `Student_31`, `Student_32`, `Student_33`, `Student_34`, `Student_35`, `Student_36`, `Student_37`, `Student_38`, `Student_39`, `Student_4`, `Student_40`, `Student_41`, `Student_42`, `Student_43`, `Student_44`, `Student_45`, `Student_46`, `Student_47`, `Student_48`, `Student_49`, `Student_5`, `Student_50`, `Student_6`, `Student_7`, `Student_8`, `Student_9`

## 4. Question ID Coverage

| Metric | Value |
|---|---|
| Unique QIDs in CSV | 12 |
| Valid QIDs (Q21–Q35) | 12 |
| Invalid / unexpected QIDs | 0 |

**QIDs found:** `Q21`, `Q22`, `Q23`, `Q24`, `Q25`, `Q26`, `Q27`, `Q28`, `Q29`, `Q30`, `Q32`, `Q34`

## 5. Student–Question Pair Counts

| Metric | Count |
|---|---|
| Total rows | 15 |
| Valid pairs (QID in Q21–Q35) | 15 |
| Invalid QID rows | 0 |
| Non-empty valid pairs | 15 |
| Empty / blank answer_text | 0 |

## 6. Duplicate Pairs

> [!NOTE]
> ✅ No duplicate (student_id, qid) pairs found.

## 7. Empty Answer Text

> [!NOTE]
> ✅ No empty answer_text values in valid pairs.

## 8. OCR Confidence Distribution

| Statistic | Value |
|---|---|
| Count | 15 |
| Mean | 0.5766 |
| Std Dev | 0.1024 |
| Min | 0.2708 |
| 25th pct | 0.5454 |
| Median | 0.6134 |
| 75th pct | 0.6351 |
| Max | 0.6869 |

**Confidence bands:**
- 🔴 Low  (< 0.40): 6.7%
- 🟡 Mid  (0.40–0.70): 93.3%
- 🟢 High (≥ 0.70): 0.0%

**Histogram (0.0 → 1.0):**
```
  [0.0–0.2]   0
  [0.2–0.4]  █ 1
  [0.4–0.6]  █████ 5
  [0.6–0.8]  █████████ 9
  [0.8–1.0]   0
```

## 9. Coverage Analysis

| Metric | Value |
|---|---|
| Theoretical max pairs (50 students × 15 Qs) | 750 |
| Ground-truth attempted (from teacher CSV) | 568 |
| Non-empty valid pairs extracted | 15 |
| **Coverage vs theoretical max** | **2.0%** |
| **Coverage vs actually attempted** | **2.6%** |

**Per-question segmented answer count (non-empty):**

| QID | Students Segmented | Coverage (of 50) |
|---|---|---|
| Q21 | 1 | 2% ⚠️ |
| Q22 | 1 | 2% ⚠️ |
| Q23 | 1 | 2% ⚠️ |
| Q24 | 2 | 4% ⚠️ |
| Q25 | 1 | 2% ⚠️ |
| Q26 | 1 | 2% ⚠️ |
| Q27 | 1 | 2% ⚠️ |
| Q28 | 1 | 2% ⚠️ |
| Q29 | 2 | 4% ⚠️ |
| Q30 | 1 | 2% ⚠️ |
| Q31 | 0 | 0% ⚠️ |
| Q32 | 1 | 2% ⚠️ |
| Q33 | 0 | 0% ⚠️ |
| Q34 | 2 | 4% ⚠️ |
| Q35 | 0 | 0% ⚠️ |

## 10. Incorrectly Segmented Question IDs

> [!NOTE]
> ✅ No incorrectly segmented QIDs found.

## 11. Sample Extracted Answers (Anonymised)

| Student | QID | Answer Preview (120 chars) | Conf |
|---|---|---|---|
| S-1 | Q21 | i,5 Precict Vakt able It USed Lable to dctaset on Vari cble (taget ) (teature ) output 1pef int6 Fror L that Megression  | 0.632 |
| S-2 | Q22 | , On Leller 1}needs This model doesn'} have   hrqh accuracy SO Fraintng Hhan dala sel . Ho Htraining dkaloso} More one O | 0.623 |
| S-2 | Q23 | 68 regressi on Oulpu} leorning   algorilhm hypex = 'poramelers 02 machince Yew Name | 0.528 |
| S-2 | Q24 | Eluaddte any 2 learn Ing Hard Kind dala se} to 6 Work (Datase} On are Con Hdentai\|. and expensi Ve ge} accuracy to In or | 0.639 |
| S-1 | Q25 | Hlosprtaltties Agrcalture Distin) terms bel ween the 96_ uish dcxta Semr Structure & Struct Urod dctc 5 0f Oksenizcti Or | 0.601 |

## 12. Recommendation

> [!CAUTION]
> ### ❌ Manual Transcription Fallback Recommended
>
> **Coverage is 2.0%** (threshold: 60%).
> **6.7% of answers have confidence < 0.40** — text is likely unreadable.
>
> **Recommended actions:**
> 1. Do not use OCR output as the sole source of answer text.
> 2. Flag all rows with `ocr_conf < 0.60` for human review.
> 3. Consider manual transcription for a representative subset to validate rubric scoring.
> 4. BA should investigate page-layout assumptions: question markers may be absent or unreadable in some PDFs.
> 5. As a fallback, use **teacher marks CSV** (already available) as the ground-truth label source for model training.

## 13. Known OCR Limitations

- Handwritten cursive text is inherently difficult for EasyOCR (trained mostly on printed text).
- Question number markers (e.g. `21.`, `Q21`) may be written differently or absent entirely.
- OCR may misread `21` as `2l`, `'` as `l`, splitting or merging tokens incorrectly.
- Multi-page answer sheets may have question text split across pages.
- Low-confidence segments must **never** be assigned a zero mark; treat as `NaN` / 'review required'.
- These limitations are intrinsic to the dataset, not a defect in BA's pipeline design.

---
*Report generated by `src/answer_feedback/validate_ocr.py` at 2026-11-11 11:47*