# MLops_T7

An MLOps course project for an AI-powered student study assistant. It combines
hobby-based learning recommendations with teacher-reviewed feedback on short
answers. Model outputs are estimates; teachers remain responsible for grades.

## Architecture

```text
Client
  -> FastAPI (`src/api`)
      -> shared request schemas (`src/common`)
      -> hobby-learning predictor (AT)
      -> answer-feedback predictor (AV)
      -> PDF rendering and OCR utilities (BA)
  -> committed model artifacts and metadata (`models/`, when supplied by AT/AV)
```

The API currently returns `503` for unavailable model integrations or missing
model artifacts; it does not fabricate predictions or model versions. The
model-owned prediction modules and artifacts must be supplied before the app
can become ready. No raw datasets or student PDFs are included in this repo.

## Local setup

Use Python 3.11 from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

To install only the API and test dependencies, use
`python -m pip install -r requirements-serving.txt`. EasyOCR is a development
dependency; its model weights are fetched by EasyOCR when first used and are
not stored in this repository.

## Run and test

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
python -m pytest -q
```

Swagger UI is at `http://127.0.0.1:8000/docs`. The API provides:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/ready` | Checks both model files and metadata |
| GET | `/api/v1/models` | Reports model metadata and versions |
| POST | `/api/v1/hobby-learning/recommend` | Hobby-based recommendation |
| POST | `/api/v1/answer-feedback/evaluate` | Per-question answer evaluation |
| POST | `/api/v1/answer-feedback/approve` | Teacher approval or mark override |

JSON request bodies are the supported API input. Each response includes a
request ID; prediction/approval responses include model name and version.
Student answer text is not written to application logs. Blank, very short, or
low-confidence OCR answers are flagged for human review.

## Docker

```powershell
docker build -t study-assistant .
docker run --rm -p 8000:8000 study-assistant
```

The image uses Python 3.11, a non-root user, and the lightweight serving
requirements. `.dockerignore` excludes raw/interim datasets, PDFs, local
environments, secrets, and MLflow databases. Until model files and metadata
are committed, `/health` works and `/ready` correctly returns `503`.

## DVC and MLflow

DVC is initialized and its two local remotes are configured in `.dvc/config`.
The root `dvc.yaml` has no stages yet: AT/AV data, preprocessing and training
pipelines have not been delivered, and this workstream does not download or
track their datasets. Run `dvc repro` after those pipeline stages are
integrated. Never add raw datasets or `*.db` files to Git.

The PRD specifies separate local MLflow tracking stores:

```text
sqlite:///mlflow_A.db  (hobby_learning)
sqlite:///mlflow_B.db  (answer_feedback)
```

For example, start the UI for one experiment store with:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow_A.db
```

MLflow is a development dependency, not a serving dependency. Registry versions
reported by the API must come from the model metadata produced by the training
workstreams.

## Data sources and limitations

- Hobby learning: MOOCCubeX, THU-KEG / XuetangX. Cite Yu et al., “MOOCCubeX:
  A Large Knowledge-centered Repository for Adaptive Learning in MOOCs,”
  CIKM 2021. Its slice is Chinese-language MOOC data and is not representative
  of all learners.
- Answer feedback: Dinesh K P, *A Dataset of Digitized Student Examination
  Papers, Answer Keys, and Manual Evaluations for Automated Grading Research*,
  Mendeley Data v1 (2026), DOI
  [10.17632/sf3kvjwknt.1](https://data.mendeley.com/datasets/sf3kvjwknt/1),
  CC BY 4.0. Do not use `/Corrected_Pdf/` as model input.

OCR quality depends on handwriting and transcription quality. Low OCR
confidence, unknown questions, and blank or short answers need human review.
The hobby-to-music analogy is a hand-curated mapping, not a learned capability.
Teacher approval returns the estimate and override; durable approval storage
is not part of this sprint implementation.
No model metrics are reported here until the model workstreams provide
verifiable training outputs.

## Team ownership

- BA: shared schemas and API, OCR/document utilities, API tests, Docker, CI,
  serving/development configuration, and this README.
- AT: `src/hobby_learning/`, its data/tests, and `docs/readme_A.md`.
- AV: answer-feedback model code (except BA's OCR/document utilities), its
  data/tests, and `docs/readme_B.md`.

Coordinate interface changes before merging model integrations. Do not commit
secrets, `.env` files, raw data, student PDFs, or MLflow database files.
