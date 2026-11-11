# AI Study Assistant - UI Application & Components

A modular, responsive web interface for interacting with and demonstrating the **AI-Powered Student Study Assistant** MLOps project.

## Architecture & Components

The UI is built with standard web technologies (zero external dependencies) and structured into reusable ES modules inside `app/components/`:

```text
app/
├── index.html                 # Single-page dashboard application container
├── styles.css                 # Responsive stylesheet with design tokens
├── app.js                     # Root controller & component orchestrator
├── server.py                  # Standalone local HTTP static server (port 3000)
├── README.md                  # Component documentation
└── components/
    ├── Header.js              # Navbar, branding & live /ready /health poller
    ├── Tabs.js                # Top-level module tab navigation
    ├── HobbyRecommender.js    # Interface for /api/v1/hobby-learning/recommend
    ├── AnswerEvaluator.js     # Interface for /api/v1/answer-feedback/evaluate
    ├── TeacherReview.js       # Interface for /api/v1/answer-feedback/approve
    └── ModelInspector.js      # Diagnostics view for /api/v1/models and lineage
```

## Component Overview

1. **Header Component (`Header.js`)**
   - Displays real-time API connectivity status (`/ready`).
   - Polls backend periodically and displays operational readiness.

2. **Hobby Recommender Component (`HobbyRecommender.js`)**
   - Submits student learning queries to `POST /api/v1/hobby-learning/recommend`.
   - Visualizes success probability via animated meter.
   - Shows weak concept badges, curated domain analogies (Music, Gaming, Sports), and cold-start notifications.

3. **Answer Evaluator Component (`AnswerEvaluator.js`)**
   - Submits segmented OCR answers to `POST /api/v1/answer-feedback/evaluate`.
   - Displays AI-estimated marks vs max marks (0.0 to 2.0).
   - Breaks down matched rubric points and missing points.
   - Triggers mandatory human review alerts for low OCR confidence (< 0.60) or short/blank answers.
   - One-click handoff button to forward evaluated answers into the Teacher Review panel.

4. **Teacher Review Component (`TeacherReview.js`)**
   - Implements human-in-the-loop review and grade overrides via `POST /api/v1/answer-feedback/approve`.
   - Records audit trail with request ID, approved score, and teacher notes.

5. **Model Inspector Component (`ModelInspector.js`)**
   - Fetches live metadata and cross-validation metrics from `/api/v1/models`.
   - Displays champion model cards for both Track A (`study-assistant-learning-model`) and Track B (`study-assistant-answer-scoring-model`).

## How to Run the UI

### Option 1: Using the included Python runner
From the repository root:
```powershell
python app/server.py
```
Then open [http://localhost:3000](http://localhost:3000) in any web browser.

### Option 2: Opening directly
Open [app/index.html](file:///c:/Users/bahul/.college/MlOps/New%20folder/MLops_T7/app/index.html) in your browser. If the backend is running (`uvicorn src.api.main:app`), it will connect directly; if offline, components provide simulated fallback responses for demonstration.

