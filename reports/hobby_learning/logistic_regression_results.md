# Model Comparison: Majority Baseline vs. Logistic Regression

**Project**: AI-Powered Student Study Assistant MLOps  
**Feature**: Feature A (Hobby Learning / Exercise Correctness Prediction)  
**Dataset**: ASSISTments 2009–2010 Corrected (Chronological Split: 80% Train, 20% Test)

## Metric Comparison

| Metric | Majority Baseline | Logistic Regression | Delta |
|---|---|---|---|
| Accuracy | 0.6915 | 0.7106 | +0.0191 |
| Precision | 0.6915 | 0.7221 | +0.0307 |
| Recall | 1.0000 | 0.9450 | -0.0550 |
| F1-score | 0.8176 | 0.8187 | +0.0011 |
| ROC-AUC | 0.5000 | 0.6590 | +0.1590 |
| PR-AUC | 0.6915 | 0.7907 | +0.0992 |

## Analysis of Results

1. **Discriminative Capacity (ROC-AUC & PR-AUC)**:
   - The majority baseline has an uninformative ROC-AUC of `0.5000` because it predicts all instances as class 1 without rank discrimination.
   - Logistic Regression achieves an ROC-AUC of `0.6590` and a PR-AUC of `0.7907` (+0.1590 gain in ROC-AUC), proving meaningful discriminative learning on historical learner performance and skill features.

2. **Precision vs. Recall Trade-off**:
   - The majority baseline achieves trivial 100% recall (`1.0000`) by predicting positive for every problem, at the cost of `0%` specificity on incorrect answers.
   - Logistic Regression improves precision to `0.7221` (++0.0307) while maintaining high recall (`0.9450`), resulting in an improved F1-score of `0.8187`.
   - Crucially, Logistic Regression successfully identifies negative instances (`correct=0`), making it substantially more useful as a diagnostic student assistant than a naïve constant predictor.

3. **Accuracy Context**:
   - Accuracy increased from `0.6915` to `0.7106`. As emphasized in the evaluation criteria, higher accuracy alone does not justify model adoption; the true improvement lies in probability ranking and positive/negative class separation reflected in ROC-AUC and PR-AUC.
