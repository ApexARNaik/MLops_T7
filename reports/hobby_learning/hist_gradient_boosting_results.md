# Model Comparison: Majority Baseline vs. Logistic Regression vs. HistGradientBoosting

**Project**: AI-Powered Student Study Assistant MLOps  
**Feature**: Feature A (Hobby Learning / Exercise Correctness Prediction)  
**Dataset**: ASSISTments 2009–2010 Corrected (Chronological Split: 80% Train, 20% Test)

## Comprehensive Metric Comparison

| Metric | Majority Baseline | Logistic Regression | HistGradientBoosting | Best Model |
|---|---|---|---|---|
| **Accuracy** | 0.6915 | 0.7106 | **0.7110** | **HistGradientBoosting** |
| **Precision** | 0.6915 | 0.7221 | **0.7240** | **HistGradientBoosting** |
| **Recall** | **1.0000** | 0.9450 | 0.9407 | **Majority Baseline** (trivial 100%) / **Logistic Regression** (learned) |
| **F1-score** | 0.8176 | **0.8187** | 0.8183 | **Logistic Regression** |
| **ROC-AUC** | 0.5000 | **0.6590** | 0.6562 | **Logistic Regression** |
| **PR-AUC** | 0.6915 | **0.7907** | 0.7902 | **Logistic Regression** |

---

## Detailed Model Analysis

### 1. Which model performs best on each metric?
- **Accuracy**: **HistGradientBoosting** achieves the highest accuracy (`0.7110`), edging out Logistic Regression (`0.7106`) and the Baseline (`0.6915`).
- **Precision**: **HistGradientBoosting** achieves highest precision (`0.7240`), indicating the fewest false positive predictions when forecasting that a student will answer correctly.
- **Recall**: **Majority Baseline** achieves a trivial `1.0000` recall by unconditionally predicting class 1 for every sample. Among genuine predictive models, **Logistic Regression** achieves higher recall (`0.9450`) than HistGradientBoosting (`0.9407`).
- **F1-Score**: **Logistic Regression** achieves the top F1-score (`0.8187` vs. `0.8183`).
- **ROC-AUC & PR-AUC**: **Logistic Regression** demonstrates slightly superior rank-ordering capability with an ROC-AUC of `0.6590` (vs. `0.6562` for HGB) and PR-AUC of `0.7907` (vs. `0.7902`).

### 2. Does HistGradientBoosting improve upon Logistic Regression?
- **In Precision and Accuracy**: Yes. HistGradientBoosting improves accuracy to `0.7110` (+0.0005) and precision to `0.7240` (+0.0019).
- **In Ranking and Probability Calibration**: No. Logistic Regression retains a slightly higher ROC-AUC (`0.6590` vs. `0.6562`) and PR-AUC (`0.7907` vs. `0.7902`). Both models perform closely, with Logistic Regression yielding smoother probability distributions across numerical history aggregations.

### 3. Do the models outperform the majority baseline?
- **Substantially Yes**: While the majority baseline reaches `0.6915` accuracy by ignoring class 0 entirely, its ROC-AUC is a random baseline `0.5000` with `0.0000` precision and recall on incorrect student responses.
- Both Logistic Regression and HistGradientBoosting provide genuine diagnostic learning, separating student mastery from struggle with ROC-AUC ~`0.66` and PR-AUC ~`0.79`.

### 4. Meaningful trade-offs between Precision and Recall
- **HistGradientBoosting** is slightly more conservative: it generates fewer false positive predictions (higher precision: `0.7240`), classifying more uncertain attempts as incorrect. This is advantageous for identifying students who genuinely need help before they attempt an exercise.
- **Logistic Regression** is slightly more generous: it yields higher recall (`0.9450`), capturing almost all correct outcomes while maintaining comparable overall discriminatory power.
