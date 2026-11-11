import json
import logging
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.answer_feedback import predict as feedback
from src.common.schemas import ApproveRequest, FeedbackRequest, HobbyRequest
from src.hobby_learning import predict as hobby


MODEL_DIR = Path("models")
MODEL_ARTIFACTS = {
    "learning": "learning_model.joblib",
    "answer": "answer_model.joblib",
}
MODEL_METADATA = {
    "learning": "learning_meta.json",
    "answer": "answer_meta.json",
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

app = FastAPI(
    title="AI Study Assistant",
    version="1.0.0",
    description=(
        "Provides hobby-based learning recommendations and teacher-reviewed "
        "answer feedback. Predictions are estimates, not final grades."
    ),
)


def _request_id(request: Request) -> str:
    return request.state.request_id


@app.middleware("http")
async def assign_request_id(request: Request, call_next: Any) -> Any:
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": _request_id(request)},
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    safe_errors = [
        {
            "loc": [str(part) for part in error["loc"]],
            "msg": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"detail": safe_errors, "request_id": _request_id(request)},
    )


def _load_metadata(model_key: str) -> dict[str, Any]:
    metadata_path = MODEL_DIR / MODEL_METADATA[model_key]
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.error("model_metadata_unavailable model=%s", model_key)
        raise HTTPException(
            status_code=503,
            detail=f"Metadata for the {model_key} model is unavailable.",
        ) from exc

    if (
        not isinstance(metadata, dict)
        or not isinstance(metadata.get("model_name"), str)
        or not metadata["model_name"].strip()
        or not isinstance(metadata.get("model_version"), (str, int))
        or isinstance(metadata.get("model_version"), bool)
        or not str(metadata["model_version"]).strip()
    ):
        log.error("model_metadata_invalid model=%s", model_key)
        raise HTTPException(
            status_code=503,
            detail=f"Metadata for the {model_key} model is invalid.",
        )
    metadata["model_version"] = str(metadata["model_version"])
    return metadata


def _model_readiness() -> list[str]:
    missing: list[str] = []
    for model_key, artifact_name in MODEL_ARTIFACTS.items():
        artifact_path = MODEL_DIR / artifact_name
        metadata_path = MODEL_DIR / MODEL_METADATA[model_key]
        if not artifact_path.is_file():
            missing.append(artifact_name)
        if not metadata_path.is_file():
            missing.append(MODEL_METADATA[model_key])
        elif artifact_path.is_file():
            _load_metadata(model_key)
    return missing


def _prediction_envelope(
    request: Request,
    result: dict[str, Any],
    model_key: str,
) -> dict[str, Any]:
    if "data" in result:
        returned_data = result["data"]
    else:
        returned_data = {
            key: value
            for key, value in result.items()
            if key not in {"model_name", "model_version", "warnings"}
        }
    if not isinstance(returned_data, dict):
        raise HTTPException(
            status_code=502,
            detail="The prediction integration returned an invalid response.",
        )

    model_name = result.get("model_name")
    model_version = result.get("model_version")
    if not isinstance(model_name, str) or not model_name.strip() or model_version is None:
        metadata = _load_metadata(model_key)
        model_name = model_name or metadata["model_name"]
        model_version = model_version or metadata["model_version"]
    if (
        not isinstance(model_name, str)
        or not model_name.strip()
        or not isinstance(model_version, (str, int))
        or isinstance(model_version, bool)
        or not str(model_version).strip()
    ):
        raise HTTPException(
            status_code=502,
            detail="The prediction integration did not report a model version.",
        )

    warnings = result.get("warnings", [])
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        raise HTTPException(
            status_code=502,
            detail="The prediction integration returned invalid warnings.",
        )

    log.info(
        "request_id=%s model=%s version=%s",
        _request_id(request),
        model_name,
        model_version,
    )
    return {
        "request_id": _request_id(request),
        "model_name": model_name,
        "model_version": str(model_version),
        "warnings": warnings,
        "data": returned_data,
    }


def _call_predictor(
    request: Request,
    module: ModuleType,
    function_name: str,
    model_key: str,
    *args: Any,
) -> dict[str, Any]:
    predictor = getattr(module, function_name, None)
    if not callable(predictor):
        log.error("prediction_integration_unavailable model=%s", model_key)
        raise HTTPException(
            status_code=503,
            detail=f"The {model_key} prediction integration is unavailable.",
        )
    try:
        result = predictor(*args)
    except ValueError as exc:
        log.warning(
            "request_id=%s prediction_rejected model=%s",
            _request_id(request),
            model_key,
        )
        raise HTTPException(
            status_code=422,
            detail="The prediction request could not be processed; check its identifiers and fields.",
        ) from exc
    except (FileNotFoundError, OSError) as exc:
        log.error("prediction_model_unavailable model=%s", model_key)
        raise HTTPException(
            status_code=503,
            detail=f"The {model_key} model is unavailable.",
        ) from exc

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=502,
            detail="The prediction integration returned an invalid response.",
        )
    return _prediction_envelope(request, result, model_key)


def _flag_uncertain_answer_results(
    data: dict[str, Any],
    answers: dict[str, str],
    confidences: dict[str, float] | None,
) -> None:
    question_results: list[dict[str, Any]] = []
    for key in ("results", "answers"):
        value = data.get(key)
        if isinstance(value, list):
            question_results.extend(
                entry for entry in value if isinstance(entry, dict)
            )
    for key, value in data.items():
        if key in answers and isinstance(value, dict):
            value.setdefault("qid", key)
            question_results.append(value)
    for result in question_results:
        qid = result.get("qid")
        if not isinstance(qid, str):
            continue
        answer = answers.get(qid)
        confidence = confidences.get(qid) if confidences is not None else None
        if answer is None or len(answer.split()) < 3 or (
            confidence is not None and confidence < 0.6
        ):
            result["human_review_required"] = True


@app.get("/health", tags=["operations"])
def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "request_id": _request_id(request)}


@app.get("/ready", tags=["operations"])
def ready(request: Request) -> dict[str, Any]:
    missing = _model_readiness()
    if missing:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "missing": missing},
        )
    metadata = [_load_metadata(key) for key in MODEL_ARTIFACTS]
    return {
        "status": "ready",
        "request_id": _request_id(request),
        "models": [
            {
                "model_name": model["model_name"],
                "model_version": model["model_version"],
            }
            for model in metadata
        ],
    }


@app.get("/api/v1/models", tags=["models"])
def models(request: Request) -> dict[str, Any]:
    return {
        "request_id": _request_id(request),
        "models": {
            "learning": _load_metadata("learning"),
            "answer": _load_metadata("answer"),
        },
    }


@app.post("/api/v1/hobby-learning/recommend", tags=["hobby learning"])
def recommend(request: Request, payload: HobbyRequest) -> dict[str, Any]:
    return _call_predictor(
        request,
        hobby,
        "recommend",
        "learning",
        payload.model_dump(),
    )


@app.post("/api/v1/answer-feedback/evaluate", tags=["answer feedback"])
def evaluate(request: Request, payload: FeedbackRequest) -> dict[str, Any]:
    response = _call_predictor(
        request,
        feedback,
        "evaluate",
        "answer",
        payload.answers,
        payload.ocr_confidence,
    )
    _flag_uncertain_answer_results(
        response["data"],
        payload.answers,
        payload.ocr_confidence,
    )
    return response


@app.post("/api/v1/answer-feedback/approve", tags=["answer feedback"])
def approve(request: Request, payload: ApproveRequest) -> dict[str, Any]:
    metadata = _load_metadata("answer")
    log.info(
        "request_id=%s model=%s version=%s teacher_approval=true",
        _request_id(request),
        metadata["model_name"],
        metadata["model_version"],
    )
    return {
        "request_id": _request_id(request),
        "assessment_id": payload.assessment_id,
        "qid": payload.qid,
        "estimated_mark": payload.estimated_mark,
        "approved_mark": payload.approved_mark,
        "max_mark": payload.max_mark,
        "status": "teacher_approved",
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
    }