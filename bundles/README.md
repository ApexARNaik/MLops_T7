# Production Model & Artifact Deployment Bundle

This directory contains the self-contained deployment bundle for the **AI-Powered Student Study Assistant** service. All trained models, metadata, schemas, calibration statistics, and evaluation metrics required for offline Docker and cloud deployment are bundled here.

## Bundle Structure

```text
bundles/
├── manifest.json                  # Complete manifest with SHA-256 checksums & lineage
├── README.md                      # Deployment instructions and overview
│
├── [Root Files for Direct Mounting / Flat Loaders]
│   ├── answer_model.joblib        # Champion model for short-answer scoring (5-fold CV trained)
│   ├── answer_meta.json           # Model metadata, rubric version hash, and CV evaluation metrics
│   ├── learning_model.joblib      # Champion model for hobby-based learning probability prediction
│   ├── learning_meta.json         # Model metadata, feature definitions, and classification metrics
│   ├── hobby_map.json             # Validated domain analogy mappings (Music, Gaming, Sports)
│   ├── problem_stats.csv          # Problem difficulty & pass-rate lookup table
│   ├── concept_stats.csv          # Concept difficulty & error-rate lookup table
│   ├── demo_learners.csv          # Historical learner profile benchmarks
│   ├── rubric.json                # Structured 15-question exam rubric & scoring key
│   ├── model_comparison_results.csv # 5-fold cross-validation comparison table
│   └── per_question_mae.csv       # Granular per-question error breakdown
│
└── [Organized Directory Structure for Mirrored Deployment]
    ├── models/                    # Mirrors the root `models/` folder expected by FastAPI
    │   ├── answer_model.joblib
    │   ├── answer_meta.json
    │   ├── learning_model.joblib
    │   ├── learning_meta.json
    │   ├── hobby_map.json
    │   ├── problem_stats.csv
    │   ├── concept_stats.csv
    │   └── demo_learners.csv
    ├── data/                      # Mirrors `data/answer_feedback/processed/`
    │   └── rubric.json
    └── docs/                      # Model governance and comparison docs
        ├── model_comparison_results.csv
        └── per_question_mae.csv
```

## Deployed Models Summary

| Model Identifier | Key / Route | File Binary | Framework | Version | Metrics Summary |
|---|---|---|---|---|---|
| `study-assistant-answer-scoring-model` | `answer` (`/api/v1/answer-feedback/evaluate`) | `answer_model.joblib` | scikit-learn Ridge / TF-IDF | `1.0.0` | MAE: 0.5365 ± 0.046, Within-0.5 Acc: 82.7%, Within-1.0 Acc: 93.1% |
| `study-assistant-learning-model` | `learning` (`/api/v1/hobby-learning/recommend`) | `learning_model.joblib` | scikit-learn Pipeline (Scaler + Regressor) | `1.0.0` | ROC-AUC: 0.875, F1: 0.833, Precision: 0.857 |

## Verification & Deployment Check

To verify readiness when deploying this bundle:

```powershell
# In Docker or local environment:
curl http://127.0.0.1:8000/health
# => {"status": "ok", "request_id": "..."}

curl http://127.0.0.1:8000/ready
# => {"status": "ready", "models": [{"model_name": "study-assistant-learning-model", ...}, {"model_name": "baseline_tfidf_ridge", ...}]}
```

