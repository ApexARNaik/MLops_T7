# AI-Powered Student Study Assistant: PRD v2.0 (3-Hour Sprint)

| Field | Value |
|---|---|
| Course | MLOps, RV College of Engineering |
| Version | **v2.0, 3-hour execution plan** (supersedes v1.0) |
| Date | 11 Nov 2026 |
| Team | AT (1), AV (2), BA (3). The 4th member is not contributing; nothing below depends on them. |
| Scope | Phase 1 (Development & Reproducibility) + Phase 2 (Productionization & Deployment). Phase 3 deferred. |

**READ THIS FIRST (teammates):** You have 3 hours total. Everything in this document is sized for that. Do not add scope. Do not wait for anyone's approval. Your section is Section 6 (AT), 7 (AV) or 8 (BA). Section 3 is the shared clock and rules; Section 5 is the interface everyone must obey.

---

## 0. What changed from v1.0

| v1.0 | v2.0 | Why |
|---|---|---|
| Faculty approval of datasets, pilot, team size | **Removed** | No time. Dataset attribution kept (one line in README). |
| Two workstreams | **Three:** A (AT), B (AV), Platform (BA) | Three active people; MLOps tooling is 15/50 marks and needs an owner. |
| Download MOOCCubeX tables, incl. 21 GB `user-problem.json` | **Stream only the first ~200 MB** of that file; skip `problem.json` (1.2 GB) and `concept.json` | Full file is infeasible in 3 hours. |
| Sentence-embedding model for Feature B | **TF-IDF + rubric-coverage features, scikit-learn only** (embeddings = optional experiment, not shipped) | Embedding libraries (torch) bloat the Docker image and slow CI. |
| Manual transcription gold set, OCR benchmark, OCR accuracy audit | **Time-boxed OCR ladder (Section 7.4)** with a hard decision at T+1:10 | OCR is the critical path; it must not eat the sprint. |
| Dataset cards, EDA reports, model cards as separate files | **One README section per feature** (source, licence, counts, limits, metrics) | Cuts paperwork, keeps the evidence. |
| Time-respecting evaluation, calibration, per-concept slices, prerequisite graph | **Cut** | Optional if time remains. |
| LLM feedback generator | **Template-based feedback from the answer key** | Deterministic, testable, no API keys. |
| Multipart PDF upload as primary API input | **JSON `answers` payload is the must-have;** PDF upload is a stretch | Keeps Docker light; demo is reliable. |
| Phase 3, HG-Bench, KILT, UI | **Out** | Not needed for the rubric in 3 hours. |
| Leakage rule: "exclude target-derived fields" | **Explicit:** the current row's `score` and `attempts` are outcome fields and must never be used as features (Section 6) | These are the most likely accidental leak in Feature A. |

---

## 1. Product in one paragraph

A study assistant with two capabilities, served by one FastAPI app:

- **A. Hobby-based learning:** a model predicts whether a learner will answer a given exercise correctly (MOOCCubeX); a separate hand-curated map turns a maths concept into a music analogy. The ML model does **not** learn hobbies. Never claim it does.
- **B. Answer-sheet feedback:** from handwritten short answers (Mendeley exam dataset) we extract text, estimate the 0-2 mark against the answer key, and explain which rubric points were missing. A teacher can approve or override. Low OCR confidence is flagged for review, never silently scored.

Core principle: estimates and feedback, not final grades. The teacher decides.

## 2. Scope and rubric mapping

**In scope (must):** DVC for data, MLflow tracking + registry for both models, baselines and leakage-aware (grouped) evaluation, PyTest, FastAPI with both endpoints, Docker, GitHub Actions (test + build + smoke test), model version in every response.
**Should:** public deployment (Section 8.6), PDF upload endpoint.
**Out:** Phase 3 (drift, Grafana, retraining), UI, HG-Bench, KILT, LLM generation.

| Rubric (marks) | Where evidence comes from |
|---|---|
| Problem and model identification (5) | Section 1 + README |
| Model design, development, evaluation (10) | MLflow runs, baseline vs candidate tables, grouped splits |
| MLOps tools (15) | DVC, MLflow registry, PyTest, FastAPI, Docker, Actions |
| Presentation and reporting (10) | README + 5-minute demo (Section 9) |
| Testing and validation (10) | PyTest report, CI run, failure-case demo |

## 3. Shared clock and ground rules

**T+0:00 = the moment BA posts "go". All times are offsets.** Hard checkpoints: if you are behind, apply your fallback (listed in your section). Do not slip silently; message the group.

| Time | AT (Feature A) | AV (Feature B) | BA (Platform) |
|---|---|---|---|
| 0:00-0:20 | Install env; start data download (6.1) | Download Mendeley zip; inspect files (7.1) | **Create repo + skeleton + stubs; push; post URL (8.1)** |
| 0:20-0:50 | Load sample, join concepts, build events + features (6.2-6.3) | Parse key/rubric, marks baseline (7.2-7.3) | Start OCR pipeline for AV (7.4/8.2) |
| 0:50-1:20 | Train baselines + model in MLflow, register (6.4) | OCR text arrives; segment answers; features (7.5) | Finish OCR; hand text to AV by **1:10** |
| 1:20-1:45 | Hobby map, `predict.py`, export model (6.5-6.6) | Train, evaluate, register, export (7.6) | FastAPI app, `/health`, `/ready`, `/models` (8.3) |
| **1:45** | **INTEGRATION DROP: push `predict.py` + `models/` + tests** | **INTEGRATION DROP** | Wire routers to real `predict()` |
| 1:45-2:20 | PyTest for A; DVC pipeline file; README section | PyTest for B; DVC pipeline file; README section | API tests, Dockerfile, build/run locally |
| 2:20-2:45 | Help BA: fix whatever fails in CI; screenshots | Same | GitHub Actions; deployment attempt (time-box 25 min) |
| 2:45-3:00 | Rehearse demo (Section 9); final README polish | Same | Final tag, confirm CI green, collect screenshots |

**Git rules (so three people do not collide):**
- One repo, owned by BA. Push straight to `main` for speed; run `git pull --rebase` before every push; commit small and often.
- **Each person edits only their own folders** (below). Shared files (`requirements*.txt`, `README.md`) are edited by BA only; others send BA lines to add, or append to their own README section file (`docs/readme_A.md`, `docs/readme_B.md`) which BA concatenates.
- Exception: for the demo of "CI blocks a failing change", BA will open one deliberate pull request at T+2:30.

| Owner | Folders / files |
|---|---|
| AT | `src/hobby_learning/`, `data/hobby_learning/`, `tests/hobby_learning/`, `docs/readme_A.md` |
| AV | `src/answer_feedback/` (except `ocr.py`, `document.py`), `data/answer_feedback/`, `tests/answer_feedback/`, `docs/readme_B.md` |
| BA | `src/common/`, `src/api/`, `src/answer_feedback/ocr.py` + `document.py`, `tests/api/`, `docker/` or `Dockerfile`, `.github/`, `README.md`, `requirements*.txt` |

**Conventions (all):** Python 3.11, seed 42, `python -m src.x.y` run from repo root, MLflow tracking URI `sqlite:///mlflow_<A|B>.db` (separate file per feature), experiment names `hobby_learning` and `answer_feedback`, registered model names `study-assistant-learning-model` and `study-assistant-answer-scoring-model`. Never commit raw data, `.env`, or `*.db` files. Never log raw student answer text.

## 4. Datasets and links (verified 11 Nov 2026)

### 4.1 Feature A: MOOCCubeX (Tsinghua KEG / XuetangX)
- Repository + docs: https://github.com/THU-KEG/MOOCCubeX
- Field docs for the exercise log: https://github.com/THU-KEG/MOOCCubeX/blob/main/docs/user-en.md
- Paper: https://lfs.aminer.cn/misc/moocdata/publications/mooccubex.pdf (Yu et al., CIKM 2021)

Direct files (base `https://lfs.aminer.cn/misc/moocdata/data/mooccube2`):

| File | Size | Use |
|---|---|---|
| `relations/user-problem.json` | **21 GB, do NOT download fully** | Learner-exercise log. Stream the first ~200 MB only. Documented fields: `log_id, user_id, problem_id, is_correct, attempts, score, submit_time`. |
| `relations/concept-problem.txt` | 1.3 MB | Problem to concept links (concept IDs embed a Chinese name and a field suffix). |
| `prerequisites/math.json` | ~59 MB | Optional. Skip unless time remains. |
| `entities/problem.json` | 1.2 GB | **Skip.** |
| `entities/concept.json` | 156 MB | **Skip.** Concept names are already inside concept IDs. |

Notes: the data is from a **Chinese-language** MOOC platform, so concept names are Chinese; AT translates a handful for the hobby map. The repo code is GPL-3.0; the README states no separate data licence, so use it for coursework and cite the paper. It contains no hobby information.

### 4.2 Feature B: Digitized Student Examination Papers (Mendeley Data)
- Page: https://data.mendeley.com/datasets/sf3kvjwknt/1 (DOI 10.17632/sf3kvjwknt.1; click **Download All**)
- Licence: **CC BY 4.0** (attribute: Dinesh K P, Mendeley Data, v1, 2026)
- Contents per the page: `Question.txt`, `answerkey.txt`, `/Student_Pdf/` (50 raw handwritten PDFs), `/Corrected_Pdf/` (50 teacher-annotated PDFs), and `Teacher_manual_marks_Anonymized.csv` (marks for Q1-Q35, all 50 students). Exam: 20 MCQs (1 mark) + 15 short answers (2 marks each), 50 marks total. Student names and IDs are anonymised and visually redacted.
- **Never use `/Corrected_Pdf/` as a model input.** It contains the teacher's marks (leakage). Use it only to eyeball things.

> Field layouts below are from the official docs and dataset page. No one has opened the files yet, so **every person's first task is to run `head` on their files and adjust column names in the code to match.**

## 5. Interfaces everyone must follow

### 5.1 Python contract (BA's routers call these; AT and AV implement them)

```python
# src/hobby_learning/predict.py  (AT)
def recommend(req: dict) -> dict: ...
# src/answer_feedback/predict.py  (AV)
def evaluate(answers: dict[str, str], ocr_conf: dict[str, float] | None = None) -> dict: ...
```

Both return `{"model_name": str, "model_version": str, "warnings": list[str], "data": {...}}`. Invalid input raises `ValueError("clear message")` (router converts to HTTP 422/400). BA creates stub versions of both files in the skeleton commit; **AT and AV overwrite their stub** after pulling.

### 5.2 Response envelope (BA adds `request_id`)

```json
{"request_id": "uuid", "model_name": "...", "model_version": "3", "warnings": [], "data": {}}
```

### 5.3 Endpoints

| Method + path | Purpose |
|---|---|
| `POST /api/v1/hobby-learning/recommend` | Body: `learner_id, subject, topic, hobby, target_problem_id` |
| `POST /api/v1/answer-feedback/evaluate` | Body: `assessment_id, answers {qid: text}, ocr_confidence {qid: float}` (optional) |
| `POST /api/v1/answer-feedback/approve` | Teacher override: `assessment_id, qid, estimated_mark, approved_mark, max_mark=2`; response shows both marks |
| `GET /health`, `GET /ready`, `GET /api/v1/models`, `GET /docs` | Liveness, readiness (model files present), active versions, Swagger |

### 5.4 Model artifacts (committed to Git so Docker and CI work offline)

`models/learning_model.joblib`, `models/learning_meta.json`, `models/answer_model.joblib`, `models/answer_meta.json`. Each `*_meta.json` holds: `model_name, model_version (registry version), mlflow_run_id, git_commit, features, metrics, trained_on, limitations`. These files are the lineage record that connects Git, DVC, MLflow and the served model.

---

## 6. Workstream A (AT): Hobby-based learning

**Target:** `is_correct` for a learner-exercise event. Baseline 0 = majority class; Baseline 1 = Logistic Regression; candidate = `HistGradientBoostingClassifier`. Grouped split by `user_id`.

### 6.1 Setup and data (0:00-0:30)

```bash
git clone <repo-url> && cd student-study-assistant-mlops
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn mlflow joblib dvc requests pytest
mkdir -p data/hobby_learning/raw data/hobby_learning/processed
```

Download the small file, then stream a slice of the large one (Python, works on Windows):

```python
# src/hobby_learning/download.py
import requests
B = "https://lfs.aminer.cn/misc/moocdata/data/mooccube2"
RAW = "data/hobby_learning/raw"
def get(url, out, limit=None):
    with requests.get(url, stream=True, timeout=60) as r, open(out, "wb") as f:
        r.raise_for_status(); n = 0
        for chunk in r.iter_content(1 << 20):
            f.write(chunk); n += len(chunk)
            if limit and n >= limit: break
get(f"{B}/relations/concept-problem.txt", f"{RAW}/concept-problem.txt")
get(f"{B}/relations/user-problem.json", f"{RAW}/user-problem-sample.jsonl", limit=200 * 1024 * 1024)
```

Then **inspect**: `head -c 1500 data/hobby_learning/raw/user-problem-sample.jsonl` and `head -5 data/hobby_learning/raw/concept-problem.txt`. Check: one JSON object per line? field names as documented? `is_correct` values (0/1, bool)? `submit_time` parseable? `concept-problem.txt` column order (assumed `problem_id<TAB>concept_id`)? If each line instead holds a nested list of logs, explode it.

Version the data (do this right after download):

```bash
dvc init                      # BA already ran this in the skeleton; skip if .dvc/ exists
dvc add data/hobby_learning/raw/user-problem-sample.jsonl data/hobby_learning/raw/concept-problem.txt
dvc remote add -d localstore ../dvc-store-A     # local folder outside the repo is fine
dvc push
git add data/hobby_learning/raw/*.dvc data/hobby_learning/raw/.gitignore .dvc/config
git commit -m "A: DVC-track MOOCCubeX sample" && git pull --rebase && git push
```

**Fallback (T+0:30):** if the sample has no usable `is_correct`, or the file will not stream, stop and tell the group. Switch Feature A to a different public knowledge-tracing dataset (for example ASSISTments) and keep the same code structure.

### 6.2 Build the events table (0:30-0:50)

```python
# src/hobby_learning/data.py
import json, pandas as pd
RAW, OUT = "data/hobby_learning/raw", "data/hobby_learning/processed"

def load_events(path=f"{RAW}/user-problem-sample.jsonl"):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: break          # last line is cut off
    return pd.DataFrame(rows)

def build_events():
    ev = load_events()
    ev["y"] = ev["is_correct"].astype(int)
    ev["t"] = pd.to_datetime(ev["submit_time"], errors="coerce")
    cp = pd.read_csv(f"{RAW}/concept-problem.txt", sep="\t", names=["problem_id", "concept_id"])
    primary = cp.groupby("problem_id").concept_id.first()
    nconc = cp.groupby("problem_id").concept_id.nunique().rename("n_concepts")
    ev = ev.merge(primary, on="problem_id").merge(nconc, on="problem_id")   # keeps only problems with a concept
    ev = ev.sort_values(["user_id", "t"]).reset_index(drop=True)            # if t is unusable: drop the sort, use file order
    g = ev.groupby("user_id")
    n = g.cumcount()
    prior = n.where(n > 0)                                                  # NaN for a user's first event
    ev["u_prev_n"] = n
    ev["u_prev_rate"] = (g["y"].cumsum() - ev["y"]) / prior
    ev["u_prev_attempts_mean"] = (g["attempts"].cumsum() - ev["attempts"]) / prior
    ev["gap_hours"] = g["t"].diff().dt.total_seconds() / 3600
    ev.to_csv(f"{OUT}/events.csv", index=False)

if __name__ == "__main__":
    build_events()
```

> **Leakage rules (write these in your README section):** (1) features use only *earlier* events of the same user (the cumsum-minus-self trick above); (2) the current row's `score`, `attempts` and `is_correct` are **never** features; (3) problem and concept difficulty are computed from **training users only** (6.3).

Quick EDA for the README (10 lines, no notebook needed): row count, number of users, number of problems, class balance of `y`, share of rows dropped because no concept link.

### 6.3 Train, evaluate, track, register (0:50-1:20)

```python
# src/hobby_learning/train.py
import json, subprocess, joblib, mlflow, pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

FEATS = ["u_prev_n", "u_prev_rate", "u_prev_attempts_mean", "gap_hours", "n_concepts", "p_rate", "c_rate"]
mlflow.set_tracking_uri("sqlite:///mlflow_A.db"); mlflow.set_experiment("hobby_learning")

ev = pd.read_csv("data/hobby_learning/processed/events.csv")
tr_i, te_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(ev, groups=ev.user_id))
tr, te = ev.iloc[tr_i].copy(), ev.iloc[te_i].copy()

gm, a = tr.y.mean(), 20                                  # smoothing strength
def smooth(col):
    s = tr.groupby(col).y.agg(["sum", "count"]); return (s["sum"] + a * gm) / (s["count"] + a)
p_rate, c_rate = smooth("problem_id"), smooth("concept_id")   # TRAIN users only
att_med = tr.u_prev_attempts_mean.median()
for d in (tr, te):
    d["p_rate"] = d.problem_id.map(p_rate).fillna(gm)
    d["c_rate"] = d.concept_id.map(c_rate).fillna(gm)
    d["u_prev_rate"] = d.u_prev_rate.fillna(gm)
    d["u_prev_attempts_mean"] = d.u_prev_attempts_mean.fillna(att_med)
    d["gap_hours"] = d.gap_hours.fillna(-1)

models = {
    "baseline_majority": DummyClassifier(strategy="most_frequent"),
    "baseline_logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "candidate_hgb": HistGradientBoostingClassifier(random_state=42),
}
best = None
for name, m in models.items():
    with mlflow.start_run(run_name=name) as run:
        m.fit(tr[FEATS], tr.y)
        pred, proba = m.predict(te[FEATS]), m.predict_proba(te[FEATS])[:, 1]
        met = {"f1": f1_score(te.y, pred), "precision": precision_score(te.y, pred),
               "recall": recall_score(te.y, pred), "roc_auc": roc_auc_score(te.y, proba)}
        mlflow.log_params({"model": name, "split": "GroupShuffleSplit(user_id) 80/20", "seed": 42,
                           "n_train": len(tr), "n_test": len(te), "features": ",".join(FEATS)})
        mlflow.log_metrics(met)
        mlflow.sklearn.log_model(m, artifact_path="model")
        if best is None or met["roc_auc"] > best["met"]["roc_auc"]:
            best = {"name": name, "m": m, "met": met, "run_id": run.info.run_id}
        print(name, met)

# register + export the winner
res = mlflow.register_model(f"runs:/{best['run_id']}/model", "study-assistant-learning-model")
MlflowClient().set_registered_model_alias("study-assistant-learning-model", "champion", res.version)
joblib.dump(best["m"], "models/learning_model.joblib")
p_rate.rename("p_rate").to_csv("models/problem_stats.csv"); c_rate.rename("c_rate").to_csv("models/concept_stats.csv")
last = ev.groupby("user_id").tail(1)[["user_id", "u_prev_n", "u_prev_rate"]].head(500)   # demo learners
last.to_csv("models/demo_learners.csv", index=False)
json.dump({"model_name": "study-assistant-learning-model", "model_version": str(res.version),
           "mlflow_run_id": best["run_id"], "chosen": best["name"], "features": FEATS, "metrics": best["met"],
           "global_rate": float(gm), "attempts_median": float(att_med),
           "git_commit": subprocess.getoutput("git rev-parse HEAD"),
           "trained_on": "first ~200 MB slice of MOOCCubeX user-problem.json",
           "limitations": "Slice not random; Chinese MOOC population; probability is an estimate."},
          open("models/learning_meta.json", "w"), indent=2)
```

Rules for the selection and honesty: if the candidate does not beat Logistic Regression, keep Logistic Regression and say so. There is no separate validation split (3-hour limit); say that in the README. Take a MLflow UI screenshot of the experiment table and of the registered-model page (`mlflow ui --backend-store-uri sqlite:///mlflow_A.db`) into `reports/figures/`.

### 6.4 The hobby map (do this in parallel while training runs; 15 min)

`models/hobby_map.json`: written by hand, reviewed by you. Steps:
1. Find the most frequent concepts among the problems in your events table that are maths (try `grep` or `str.contains("数学")` on `concept_id`; check the suffix pattern first).
2. Take the **top ~8**, translate the names to English, and write an entry for each. If a hobby analogy would be forced, set `"valid": false`. The product must be able to say "no good analogy".

```json
{
  "Music": {
    "<concept_id>": {
      "topic": "Ratios",
      "valid": true,
      "analogy": "An octave is a 2:1 frequency ratio and a perfect fifth is 3:2.",
      "worked_example": "A string at 220 Hz sounds an octave above at 440 Hz...",
      "why_valid": "Pitch intervals are defined by frequency ratios.",
      "limits": "Only covers ratio intuition, not ratio word problems."
    },
    "<other_concept_id>": {"topic": "Limits", "valid": false, "reason": "No sound analogy that is mathematically accurate."}
  }
}
```

Plausible music analogies, if those concepts appear: ratios and proportion (intervals), fractions (note lengths), periodic functions (sound waves), logarithms (equal-tempered semitones, ratio 2^(1/12)), permutations (note orderings). Use only those that match the concepts actually in the data.

### 6.5 `predict.py` behaviour (1:20-1:45)

`recommend(req)`: validate fields; `subject` must be "Mathematics" (else `ValueError`); look up the learner in `demo_learners.csv`; if unknown, use global rate and add warning `"cold start: no history for this learner"`; look up problem and concept difficulty (fallback to global rate with a warning); build the feature row in the exact `FEATS` order; return `p_correct`, `predicted_correct`, `weak_concepts` (the target problem's concept(s) with the lowest `c_rate`), `analogy` from `hobby_map.json` (or `null` plus a message when invalid or unknown hobby), and a recommendation text. **Do not return raw learner IDs beyond echoing the request.** The probability must be clamped to [0,1].

### 6.6 DVC pipeline, tests, README (1:45-2:20)

`src/hobby_learning/dvc.yaml`:

```yaml
stages:
  prepare:
    cmd: python -m src.hobby_learning.data
    deps: [src/hobby_learning/data.py, data/hobby_learning/raw/user-problem-sample.jsonl, data/hobby_learning/raw/concept-problem.txt]
    outs: [data/hobby_learning/processed/events.csv]
  train:
    cmd: python -m src.hobby_learning.train
    deps: [src/hobby_learning/train.py, data/hobby_learning/processed/events.csv]
```

(Models are intentionally not DVC `outs`; they are committed to Git so Docker/CI work.) Run `dvc repro` once and commit `dvc.lock`.

Minimum PyTest (`tests/hobby_learning/`): (1) feature test on a 6-row hand-made frame proving `u_prev_rate` for event *k* ignores event *k* and later; (2) `recommend` returns probability in [0,1] and includes `model_version`; (3) unknown learner returns the cold-start warning; (4) unknown hobby returns `analogy: null`; (5) wrong subject raises `ValueError`.

README section (`docs/readme_A.md`, about 15 lines): source + citation, what slice was used and why, counts, class balance, leakage rules, baseline vs candidate table, limitations (slice not random; Chinese MOOC population; hobby map is hand-written and not learned).

---

## 7. Workstream B (AV): Answer-sheet scoring and feedback

**Target:** teacher mark for short-answer questions (expected 0, 1 or 2; **check whether 0.5 steps exist** in the CSV). Baseline 0 = per-question mean; Baseline 1 = TF-IDF + Ridge; candidate = small tree/boosting model over engineered features. Grouped CV by student.

### 7.1 Download and inspect (0:00-0:20)

1. Open https://data.mendeley.com/datasets/sf3kvjwknt/1, click **Download All**, unzip into `data/answer_feedback/raw/`.
2. `dvc add data/answer_feedback/raw` (BA's skeleton already ran `dvc init`), then `dvc remote add -d localstore ../dvc-store-B`, `dvc push`, commit the `.dvc` file.
3. Open `Question.txt`, `answerkey.txt` and the marks CSV. Identify: which columns are Q21-Q35 (assumed short answers; **verify against `Question.txt`**), marks distribution per question, and how many PDF pages each student sheet has. Open 2-3 raw PDFs: how do students label answers (Q21, 21., none)? Is there one question per page?

### 7.2 Rubric file (0:20-0:40)

Create `data/answer_feedback/processed/rubric.json` (small, commit it): for each short-answer question `{"qid": "Q21", "max_mark": 2, "reference": "<answer key text>", "points": [["keyword", "synonym"], ["another point"]]}`. Fifteen questions with 2-4 points each takes about 15 minutes by hand from `answerkey.txt`. Each inner list is one rubric point: any phrase in the list counts as covering it.

### 7.3 Marks-only baseline (0:30-0:50, no OCR needed)

Load the marks CSV into long format `(student_id, qid, mark)` for the short-answer columns. Compute the per-question mean baseline with `GroupKFold` by student and log MAE/RMSE/exact-accuracy to MLflow now (`sqlite:///mlflow_B.db`, experiment `answer_feedback`). This gives you a working baseline run before OCR text arrives.

### 7.4 OCR ladder (BA builds the pipeline; AV owns the decision)

BA produces `data/answer_feedback/interim/answers_ocr.csv` with columns `student_id, qid, answer_text, ocr_conf, source`. Order of attempts, time-boxed:

1. **(10 min)** Render 3 pages with PyMuPDF and test **EasyOCR** on them. Handwriting is usually poor with classical OCR; judge by eye.
2. **(15 min) Vision-capable LLM transcription:** if any teammate has an API key for a vision-capable model, send each page image and request JSON `{qid: text}`. Keep the call behind the adapter `OCREngine.transcribe(image) -> (text, confidence)`; read the key from `.env` (never committed). The dataset is anonymised and CC BY 4.0.
3. **Decision at T+1:10:** if (student, question) coverage after segmentation is below about 60% or text is unreadable, **fall back to manual typing**: AV, BA and AT each type the answers for 5 short-answer questions × about 30 students (about 50 each) from the raw PDFs, with `source="manual"`, `ocr_conf=1.0`. State this plainly in the README; the scoring model is then evaluated on manual transcripts and OCR is reported as a limitation.

Segmentation heuristic: regex such as `(?m)^\s*(?:Q\.?\s*)?(\d{1,2})\s*[.)\-:]` accepting only IDs in the short-answer range in increasing order; report % of (student, question) pairs found. Anything not found is **missing, not zero**: flag for review.

### 7.5 Features (1:10-1:30)

Per answer row: `tfidf_cos` (cosine between answer and `reference`, vectorizer fit on references), `coverage` (fraction of rubric points matched; normalise to lowercase, strip punctuation, match phrases or at least 60% token overlap), `max_point_hit`, `n_words`, `log_words`, `ocr_conf`, `question_mean_mark` (from **training folds only**), `max_mark`.
**Never use anything from `/Corrected_Pdf/`.**

### 7.6 Train, evaluate, register (1:20-1:45)

- `GroupKFold(n_splits=5)` by `student_id`. Models: `baseline_question_mean`, `baseline_tfidf_ridge`, `candidate_gradient_boosting` (or RandomForest) over the features above. Round and clip predictions to the allowed marks (`0..max_mark`, step per the CSV).
- Log per model to MLflow: mean and std across folds of MAE, RMSE, exact-mark accuracy, within-0.5/within-1 accuracy; plus a per-question MAE table as an artifact.
- Pick the winner; if the candidate does not beat the baseline, ship the baseline and say so. Register `study-assistant-answer-scoring-model`, set alias `champion`, export `models/answer_model.joblib` and `models/answer_meta.json` (same fields as 5.4; include `rubric_version` = hash of `rubric.json`).

### 7.7 `predict.py` and `feedback.py` (1:30-1:45)

`evaluate(answers, ocr_conf)`: load rubric; for each `qid`: if not in the rubric, raise `ValueError` (or return an entry with `human_review_required: true` and no mark); compute features; predict; clip to `[0, max_mark]`.
Per-question output: `qid, estimated_mark, max_mark, approved_mark: null, status: "pending_teacher_review", matched_points, missing_points, deduction_rationale, improvement_suggestion, evidence_refs [{source: "answerkey.txt", qid, point}], ocr_confidence, human_review_required`.
`human_review_required = True` if `ocr_conf < 0.6`, or fewer than 3 words (incl. empty), or unknown question. **An empty or unreadable answer must never produce a confident score.**
Feedback is template-based from `missing_points` (for example "The answer does not mention: ..."; "To earn the full mark, include: ...") and states plainly that the answer key is the source. No generative claims, no invented citations.

### 7.8 DVC, tests, README (1:45-2:20)

- `src/answer_feedback/dvc.yaml` with stages `features` (OCR csv + rubric to features csv) and `train`, same pattern as 6.6. Run `dvc repro` once; commit `dvc.lock`.
- PyTest (`tests/answer_feedback/`): predicted mark always in `[0, max_mark]`; empty answer is flagged for review; low OCR confidence is flagged; missing/unknown question is handled; coverage function on a hand-made example; no feature uses the mark column.
- README section (`docs/readme_B.md`): source, CC BY attribution, counts, mark distribution, how OCR was obtained (and which rung of the ladder), baseline vs candidate table, per-question errors, limitations (one course, 50 students, handwriting quality, scoring is only as good as the transcript).

---

## 8. Workstream C (BA): Platform, OCR pipeline, API, Docker, CI/CD

### 8.1 Skeleton repo (0:00-0:20, this unblocks the others)

```bash
mkdir student-study-assistant-mlops && cd student-study-assistant-mlops
git init -b main && dvc init
mkdir -p src/{common,api,hobby_learning,answer_feedback} tests/{api,hobby_learning,answer_feedback} \
  data/{hobby_learning,answer_feedback}/{raw,interim,processed} models reports/figures docs .github/workflows
touch src/__init__.py src/{common,api,hobby_learning,answer_feedback}/__init__.py
```

`.gitignore` must include: `.venv/`, `__pycache__/`, `.env`, `*.db`, `mlruns/`, `mlartifacts/`, `data/*/raw/`, `data/*/interim/`, `*.pdf`. (`models/` and `data/*/processed/rubric.json` are committed.)
`pytest.ini`: `[pytest]` / `pythonpath = .`.
`requirements-serving.txt` (what the Docker image needs): `fastapi`, `uvicorn`, `pydantic`, `python-multipart`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `httpx` (tests), `pytest`. `requirements-dev.txt` adds `mlflow`, `dvc`, `requests`, `pymupdf`, `easyocr`. Pin exact versions with `pip freeze` once everything works (T+2:00).

`src/common/schemas.py`:

```python
from pydantic import BaseModel, Field
class HobbyRequest(BaseModel):
    learner_id: str; subject: str; topic: str; hobby: str; target_problem_id: str
class FeedbackRequest(BaseModel):
    assessment_id: str
    answers: dict[str, str]
    ocr_confidence: dict[str, float] | None = None
class ApproveRequest(BaseModel):
    assessment_id: str; qid: str
    estimated_mark: float = Field(ge=0); approved_mark: float = Field(ge=0); max_mark: float = 2
```

Also commit **stub** `src/hobby_learning/predict.py` and `src/answer_feedback/predict.py` returning a dummy envelope-shaped dict (model_version "stub"), so BA can build the API immediately. Push, add AT and AV as collaborators, post the repo URL. Create the 3 empty `docs/readme_*.md` files.

### 8.2 OCR pipeline for AV (0:20-1:10; `src/answer_feedback/ocr.py`, `document.py`)

- `pip install pymupdf easyocr`. `document.py`: render each page of a PDF to an image (`fitz.open(path)`; `page.get_pixmap(dpi=200)`); validate file type, size and page count.
- `ocr.py`: `OCREngine` protocol with `transcribe(image) -> (text, confidence)`; implement `EasyOCREngine` first; add the vision-LLM engine only if a key exists (read from env, never hard-coded).
- Script `python -m src.answer_feedback.ocr --all` runs all 50 PDFs, writes per-page text to `data/answer_feedback/interim/ocr_pages/` and the segmented `answers_ocr.csv` (see 7.4). Print the coverage number. **Deliver to AV by T+1:10**; if blocked, trigger the manual fallback in 7.4 early.

### 8.3 FastAPI app (1:10-1:45)

```python
# src/api/main.py
import json, logging, uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from src.common.schemas import HobbyRequest, FeedbackRequest, ApproveRequest
from src.hobby_learning import predict as hobby
from src.answer_feedback import predict as feedback

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")
app = FastAPI(title="AI Study Assistant", version="1.0.0")

def envelope(res: dict) -> dict:
    rid = str(uuid.uuid4())
    log.info("request_id=%s model=%s version=%s", rid, res["model_name"], res["model_version"])  # never log answer text
    return {"request_id": rid, **res}

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/ready")
def ready():
    need = ["models/learning_model.joblib", "models/answer_model.joblib"]
    missing = [p for p in need if not Path(p).exists()]
    if missing: raise HTTPException(503, f"missing model files: {missing}")
    return {"status": "ready"}

@app.get("/api/v1/models")
def models():
    return {k: json.loads(Path(f"models/{k}_meta.json").read_text()) for k in ("learning", "answer")}

@app.post("/api/v1/hobby-learning/recommend")
def recommend(req: HobbyRequest):
    try: return envelope(hobby.recommend(req.model_dump()))
    except ValueError as e: raise HTTPException(422, str(e))

@app.post("/api/v1/answer-feedback/evaluate")
def evaluate(req: FeedbackRequest):
    try: return envelope(feedback.evaluate(req.answers, req.ocr_confidence))
    except ValueError as e: raise HTTPException(422, str(e))

@app.post("/api/v1/answer-feedback/approve")
def approve(req: ApproveRequest):
    if req.approved_mark > req.max_mark: raise HTTPException(422, "approved_mark exceeds max_mark")
    return {"assessment_id": req.assessment_id, "qid": req.qid, "estimated_mark": req.estimated_mark,
            "approved_mark": req.approved_mark, "status": "teacher_approved"}
```

(Splitting into two router files is optional; do it only if everything else is done.) Add Swagger example payloads via Pydantic `model_config` / `json_schema_extra` if time permits. Run `uvicorn src.api.main:app --reload` and test in `/docs`.

### 8.4 API tests (1:45-2:10; `tests/api/test_api.py`)

Use `fastapi.testclient.TestClient`. Cover: `/health` is 200; `/ready` is 200 with real models; `/api/v1/models` lists both versions; invalid payload (missing field) returns 422; `recommend` response has `request_id`, `model_version`, probability in [0,1]; `evaluate` response has `estimated_mark <= max_mark` and flags a blank answer; `approve` rejects `approved_mark > max_mark`; `evaluate` with an unknown question returns 422 or a review flag.

### 8.5 Dockerfile and local run (2:00-2:20)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt
COPY src ./src
COPY models ./models
COPY data/answer_feedback/processed/rubric.json ./data/answer_feedback/processed/rubric.json
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Add anything else the `predict.py` files load at runtime (for example `models/hobby_map.json`, which already sits in `models/`). Run `docker build -t study-assistant . && docker run -p 8000:8000 study-assistant`, then hit `/health`, `/ready`, `/docs`. No secrets or raw data in the image. Add a `.dockerignore` (`data/*/raw`, `.venv`, `.git`, `*.db`).

### 8.6 GitHub Actions and deployment (2:10-2:50)

`.github/workflows/ci.yml`:

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements-serving.txt
      - run: pytest -q
  docker:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t study-assistant:${{ github.sha }} .
      - name: smoke test
        run: |
          docker run -d -p 8000:8000 --name sa study-assistant:${{ github.sha }}
          sleep 10
          curl -f http://localhost:8000/health
          curl -f http://localhost:8000/ready
```

`needs: test` is the gate: a failing test blocks the image build. At **T+2:30** open one deliberate pull request that breaks a test, screenshot the red check, then fix it and screenshot green. Tests must pass without raw data, DVC pulls or MLflow; they rely only on committed `models/`.

**Deployment (hard time-box 25 minutes):** deploy the Docker image from the GitHub repo to a free host you already have an account with (for example Render or a Hugging Face Docker Space; check its free-tier memory and port rules on the spot). If it is not live by T+2:50, **stop** and demo `docker run` locally. The v1.0 acceptance criteria already allow that, provided you say so clearly.

### 8.7 Final deliverables (2:45-3:00)

Consolidated `README.md`: problem, architecture diagram (the ASCII in v1.0 is fine), how to run (`docker run`, `pytest`, `dvc repro`, `mlflow ui`), links to `docs/readme_A.md` and `docs/readme_B.md`, dataset attributions, limitations, screenshots (`reports/figures/`: MLflow experiments + registry for both models, pytest output, Swagger, docker run, green and red CI). Tag `v1.0`.

---

## 9. Demo script (5 minutes, rehearse at 2:45)

1. **Data and versions (30 s):** show `.dvc` files, `dvc.lock`, dataset sources and the leakage rules.
2. **Experiments (60 s):** MLflow experiments for both features, baseline vs candidate, registered model pages with versions.
3. **Tests and CI (45 s):** `pytest` pass; red-then-green GitHub Actions screenshots.
4. **Docker + API (120 s):** `docker run`; `/docs`; one `recommend` call (probability, weak concept, music analogy, model version); one `evaluate` call with a good answer and a blank/low-OCR answer showing `human_review_required: true`; one `approve` call showing estimated vs approved mark.
5. **Honest limits (45 s):** small dataset, OCR quality (state which rung was used), hobby map is hand-written, probability is an estimate, teacher decides.

## 10. Definition of done (tick before the demo)

- [ ] Both raw datasets DVC-tracked; `dvc.lock` committed for both pipelines
- [ ] Two MLflow experiments, each with baseline and candidate runs and metrics
- [ ] Grouped split by learner (A) and student (B), documented in each README section
- [ ] Both models registered (`champion` alias) and exported to `models/` with `*_meta.json`
- [ ] PyTest passes locally and in CI (includes failure cases: empty answer, unknown learner, bad payload)
- [ ] `POST /recommend` and `POST /evaluate` work in `/docs`; responses carry `request_id` and `model_version`
- [ ] Teacher approve endpoint shows estimated vs approved mark
- [ ] Docker image builds, runs, and `/health` + `/ready` pass
- [ ] GitHub Actions: tests gate the Docker build; smoke test runs
- [ ] Deployed URL **or** documented local `docker run` demo
- [ ] README with attributions (Mendeley CC BY 4.0; MOOCCubeX paper), limits and screenshots

## 11. Risks, fallbacks and the cut list

| Risk | Fallback |
|---|---|
| MOOCCubeX slice lacks usable `is_correct` or will not stream | Switch A to another public knowledge-tracing dataset at T+0:30; keep code structure |
| First 200 MB covers few users or a skewed slice | Say so in the README; the grouped split still prevents user leakage |
| Few maths problems appear in the slice | Train on all linked problems; the hobby layer only activates for concepts with a valid map entry |
| Handwriting OCR unusable | Ladder in 7.4; manual transcripts for 5 questions × about 30 students; disclose it |
| Answer segmentation fails | Report coverage; unmatched answers are "missing", flagged for review |
| Candidate does not beat baseline | Ship the baseline and document it |
| Docker image too big or build slow | Keep `requirements-serving.txt` free of torch, mlflow, easyocr, dvc |
| CI fails because of missing files | Commit `models/`, `rubric.json`, `hobby_map.json`; never read raw data in tests |
| Deployment platform fights back | Stop at 25 minutes; demo local Docker |

**If behind, cut in this order (last item first):** UI (already out) → PDF upload endpoint → public deployment → router split → extra tests beyond the checklist → candidate model (keep baselines + MLflow + registry). **Never cut:** DVC, MLflow registry, PyTest, FastAPI endpoints, Docker, CI test + build.

## 12. Phase 3 (not in this sprint)

Drift monitoring, Grafana/Prometheus, retraining triggers and a full model card are deferred. Cheap preparation already included: `request_id` and `model_version` in every response and logs that never contain raw answers.

## 13. Sources

- **[S1]** MOOCCubeX, THU-KEG: https://github.com/THU-KEG/MOOCCubeX (docs: `docs/user-en.md`); paper: Yu et al., "MOOCCubeX: A Large Knowledge-centered Repository for Adaptive Learning in MOOCs," CIKM 2021, https://lfs.aminer.cn/misc/moocdata/publications/mooccubex.pdf
- **[S2]** Dinesh K P, "A Dataset of Digitized Student Examination Papers, Answer Keys, and Manual Evaluations for Automated Grading Research," Mendeley Data, V1, 14 April 2026, DOI 10.17632/sf3kvjwknt.1, https://data.mendeley.com/datasets/sf3kvjwknt/1 (CC BY 4.0)
- **[S0]** Faculty MLOps circular (phases, rubric): as supplied by the team.
