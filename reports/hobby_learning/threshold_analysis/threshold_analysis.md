# Feature A Threshold Selection Experiment Report: Hobby Learning / Exercise Correctness Prediction

**Project:** AI-Powered Student Study Assistant MLOps  
**Feature:** Feature A — Hobby Learning / Exercise Correctness Prediction  
**Dataset:** ASSISTments Interaction Dataset (`data/hobby_learning/processed/events.csv`)  
**Experiment Date:** November 2026  

---

## 1. Executive Summary

In the existing implementation, the binary classification threshold for predicting whether a student will correctly answer an exercise was set to the conventional default of `0.50`:

$$\text{predicted\_correct} = \mathbb{I}(p_{\text{correct}} \ge 0.50)$$

This experiment empirically evaluated whether $0.50$ is optimal or whether a tuned threshold yields superior performance on student correctness predictions. Using a strictly chronological three-way split based on `order_id` (70% train, 15% validation, 15% test) without temporal leakage:

1. **Validation Threshold Selection:**
   - Both **Logistic Regression** and **HistGradientBoosting** achieved their highest validation F1-score at threshold **`0.40`** (Logistic Regression Val F1: `0.7856` vs. `0.7773` at 0.50; HistGradientBoosting Val F1: `0.7917` vs. `0.7839` at 0.50).
2. **Untouched Test Set Evaluation:**
   - On the untouched test set, threshold `0.40` slightly increased raw F1-score (+0.0023 for LR, +0.0061 for HGB) due to higher recall (`0.9917` for LR, `0.9860` for HGB).
   - However, threshold `0.40` caused a **severe collapse in specificity and a drastic increase in false positives**:
     - For Logistic Regression, false positives surged from `13,543` (at 0.50) to `15,018` (at 0.40), dropping specificity from `13.17%` down to an unacceptable `3.72%`.
     - For HistGradientBoosting, false positives surged from `13,585` (at 0.50) to `14,760` (at 0.40), dropping specificity from `12.91%` down to `5.37%`.
3. **Serving Architecture & Recommendation:**
   - The deployed FastAPI endpoint (`/recommend/hobby`) loads `models/learning_model.joblib` and `models/learning_meta.json` (trained on a 7-feature slice), **not** the candidate ASSISTments models under `models/hobby_learning/`.
   - In educational recommender systems, **false positives are costly**: predicting that a student will solve an exercise when they actually cannot leads to recommending excessively difficult exercises, causing frustration and learning dropouts.
   - **Recommendation:** **Retain the `0.50` default threshold** for production serving and candidate benchmarks, or tune toward higher precision/balanced accuracy ($\ge 0.50 - 0.60$) if avoiding false positives is prioritized.

---

## 2. Dataset and Chronological Split Details

The experiment used the processed ASSISTments dataset:
- **Source:** `data/hobby_learning/processed/events.csv`
- **Total Interaction Records:** `346,860`
- **Target Variable:** `correct` (Binary: `1` = student answered correctly on first attempt, `0` = incorrect)
  - Class Distribution: `64.53%` Correct (`1`), `35.47%` Incorrect (`0`)
- **Order Column:** `order_id` (Monotonically increasing chronological sequence)

### Chronological Three-Way Partitioning (No Data Leakage)
To strictly avoid future-to-past data leakage and random shuffle leakage:
- **Training Set (70%):** First `242,801` rows ($\text{order\_id} \le \text{train\_max}$)
- **Validation Set (15%):** Next `52,030` rows ($\text{train\_max} < \text{order\_id} \le \text{val\_max}$)
- **Test Set (15%):** Final `52,029` rows ($\text{val\_max} < \text{order\_id}$)

All feature scaling and imputations were fitted **strictly on the training partition** and transformed forward to validation and test sets.

### Feature Definitions
The 7 anti-leakage historical features:
1. `user_prev_count`: Cumulative previous interactions by student.
2. `user_prev_accuracy`: Cumulative rolling accuracy of student.
3. `user_prev_attempts_mean`: Average attempts per previous problem.
4. `user_prev_hints_mean`: Average hints requested per previous problem.
5. `user_skill_prev_count`: Prior attempts on the current skill component.
6. `user_skill_prev_accuracy`: Prior accuracy on the current skill component.
7. `n_skill_components`: Number of skill components tagged for the exercise.

---

## 3. Models Evaluated

Two candidate classification pipelines were trained on the 70% training split:

1. **Logistic Regression Pipeline:**
   - Preprocessing: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
   - Estimator: `LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)`.
2. **HistGradientBoosting Pipeline:**
   - Preprocessing: `SimpleImputer(strategy='median')`.
   - Estimator: `HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)`.

---

## 4. Threshold Sweep Methodology

Thresholds tested:
`[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]`

For each model and threshold, the following metrics were computed on the validation split:
- Accuracy, Precision, Recall, F1-Score
- Specificity ($TN / (TN + FP)$)
- Balanced Accuracy ($(Recall + Specificity) / 2$)
- False Positive Count ($FP$) & False Negative Count ($FN$)
- Confusion Matrix

---

## 5. Validation Split Results

### 5.1 Validation Metrics Summary Across Key Criteria

| Model | Criterion | Threshold | Accuracy | Precision | Recall | F1-Score | Specificity | Balanced Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **Primary (Best F1)** | **0.40** | **0.6944** | **0.6787** | **0.9323** | **0.7856** | **0.3371** | **0.6347** |
| Logistic Regression | Baseline (Default) | 0.50 | 0.7017 | 0.7043 | 0.8671 | 0.7773 | 0.4531 | 0.6601 |
| Logistic Regression | Highest Precision | 0.85 | 0.4327 | 0.8510 | 0.0667 | 0.1238 | 0.9824 | 0.5246 |
| Logistic Regression | Highest Recall | 0.10 | 0.6048 | 0.6033 | 0.9978 | 0.7520 | 0.0144 | 0.5061 |
| Logistic Regression | Highest Balanced Acc | 0.60 | 0.6870 | 0.7278 | 0.7645 | 0.7457 | 0.5705 | 0.6675 |
| **HistGradientBoosting** | **Primary (Best F1)** | **0.40** | **0.7017** | **0.6816** | **0.9442** | **0.7917** | **0.3374** | **0.6408** |
| HistGradientBoosting | Baseline (Default) | 0.50 | 0.7095 | 0.7082 | 0.8777 | 0.7839 | 0.4568 | 0.6673 |
| HistGradientBoosting | Highest Precision | 0.90 | 0.4063 | 0.8844 | 0.0127 | 0.0251 | 0.9975 | 0.5051 |
| HistGradientBoosting | Highest Recall | 0.10 | 0.6371 | 0.6236 | 0.9977 | 0.7675 | 0.0954 | 0.5465 |
| HistGradientBoosting | Highest Balanced Acc | 0.60 | 0.6860 | 0.7494 | 0.7166 | 0.7327 | 0.6400 | 0.6783 |

---

## 6. Final Test Set Evaluation (Selected vs. 0.50 Baseline)

The selected threshold (`0.40`) was evaluated on the **untouched final test partition** (`52,029` rows) and compared against the baseline `0.50`:

| Model | Threshold Type | Threshold | Accuracy | Precision | Recall | F1-Score | Specificity | Balanced Acc | False Positives | False Negatives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Selected (Best Val F1) | **0.40** | 0.7055 | 0.7064 | 0.9917 | 0.8251 | 0.0372 | 0.5144 | 15,018 | 303 |
| **Logistic Regression** | Default Baseline | **0.50** | **0.7107** | **0.7206** | 0.9586 | 0.8227 | **0.1317** | **0.5452** | **13,543** | 1,508 |
| **HistGradientBoosting**| Selected (Best Val F1) | **0.40** | **0.7065** | 0.7088 | 0.9860 | **0.8247** | 0.0537 | 0.5198 | 14,760 | 511 |
| **HistGradientBoosting**| Default Baseline | **0.50** | 0.7047 | **0.7184** | 0.9512 | 0.8186 | **0.1291** | **0.5401** | **13,585** | 1,777 |

### Analysis of Test Set Results:
- **Did threshold 0.40 genuinely improve performance?**  
  **No, not from a balanced pedagogical standpoint.**  
  While the F1-score marginally increased (+0.0024 for LR, +0.0061 for HGB), this occurred solely by shifting almost all predictions into the positive class (Recall > 98.5%). As a consequence:
  - For Logistic Regression, the model predicted positive for 98.3% of all test instances, correctly identifying only **580** out of 15,598 incorrect attempts ($\text{Specificity} = 3.72\%$). False positives climbed by `+1,475`.
  - For HistGradientBoosting, the model correctly identified only **838** out of 15,598 incorrect attempts ($\text{Specificity} = 5.37\%$). False positives climbed by `+1,175`.
  - The baseline threshold `0.50` provided noticeably higher precision (`72.06%` vs `70.64%`), much higher specificity (`13.17%` vs `3.72%`), and superior balanced accuracy (`54.52%` vs `51.44%`).

---

## 7. Serving Implementation vs. Candidate Models

A critical architectural distinction exists in the codebase:
- **Deployed API Model:**  
  The endpoint `POST /recommend/hobby` in `src/api/main.py` invokes `src/hobby_learning/predict.py`. It explicitly loads:
  - Model artifact: `models/learning_model.joblib`
  - Metadata: `models/learning_meta.json`  
  This model uses a 7-feature schema tailored to MOOCCubeX (`u_prev_n`, `u_prev_rate`, `u_prev_attempts_mean`, `gap_hours`, `n_concepts`, `p_rate`, `c_rate`).
- **Feature A ASSISTments Candidate Models:**  
  The candidate models trained on ASSISTments are located in `models/hobby_learning/` (`logistic_regression.joblib`, `hist_gradient_boosting.joblib`).
- **Configurable Serving Threshold:**  
  `src/hobby_learning/predict.py` has been updated to dynamically read `classification_threshold` from `models/learning_meta.json` rather than hardcoding `0.5`.  
  The production model continues to execute at its established `0.50` threshold until a dedicated validation experiment is run specifically for the serving artifact.

---

## 8. Artifacts Generated

All artifacts have been generated in `reports/hobby_learning/threshold_analysis/`:
1. `threshold_comparison.csv`: Complete metrics for all 17 thresholds across both models on the validation split.
2. `test_set_comparison.csv`: Head-to-head comparison between threshold 0.40 and 0.50 on the untouched test partition.
3. `threshold_analysis.md`: This comprehensive report.
4. `threshold_vs_metrics.png`: Visual plots showing accuracy, precision, recall, and F1 across thresholds.
5. `confusion_matrices.png`: 2x2 comparison grid of test confusion matrices.
6. `threshold_results.json`: Full machine-readable experimental data.

---

## 9. Conclusion and Final Recommendation

1. **Recommendation:** **Retain the default threshold of `0.50`**.
2. **Pedagogical Rationale:** Lowering the threshold to `0.40` causes catastrophic false positive inflation (+1,100 to +1,400 false positives), essentially creating a near-majority-positive classifier that fails to identify struggling learners who need intervention.
3. **Serving Safety:** The serving endpoint `models/learning_model.joblib` remains safely configured with `0.50` without breaking production inference.

