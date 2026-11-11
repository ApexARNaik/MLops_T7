import json
import uuid

import pytest
from fastapi.testclient import TestClient

import src.api.main as api


@pytest.fixture
def client(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setattr(api, "MODEL_DIR", model_dir)
    return TestClient(api.app)


def write_model_artifacts(model_dir, *, include_models=True):
    model_details = {
        "learning": ("study-assistant-learning-model", "4"),
        "answer": ("study-assistant-answer-scoring-model", "2"),
    }
    for key, (model_name, version) in model_details.items():
        (model_dir / api.MODEL_METADATA[key]).write_text(
            json.dumps({"model_name": model_name, "model_version": version}),
            encoding="utf-8",
        )
        if include_models:
            (model_dir / api.MODEL_ARTIFACTS[key]).write_bytes(b"test artifact")


def assert_request_id(response):
    request_id = response.json()["request_id"]
    uuid.UUID(request_id)
    assert response.headers["X-Request-ID"] == request_id


def test_health_returns_status_and_request_id(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert_request_id(response)


def test_ready_reports_missing_model_artifacts(client):
    response = client.get("/ready")

    assert response.status_code == 503
    assert "learning_model.joblib" in response.json()["detail"]["missing"]
    assert_request_id(response)


def test_ready_returns_model_versions_when_artifacts_exist(client, tmp_path):
    write_model_artifacts(api.MODEL_DIR)

    response = client.get("/ready")

    assert response.status_code == 200
    assert {model["model_version"] for model in response.json()["models"]} == {"2", "4"}
    assert_request_id(response)


def test_models_endpoint_reports_both_versions(client):
    write_model_artifacts(api.MODEL_DIR, include_models=False)

    response = client.get("/api/v1/models")

    assert response.status_code == 200
    assert response.json()["models"]["learning"]["model_version"] == "4"
    assert response.json()["models"]["answer"]["model_version"] == "2"
    assert_request_id(response)


def test_models_endpoint_fails_when_metadata_is_missing(client):
    response = client.get("/api/v1/models")

    assert response.status_code == 503
    assert_request_id(response)


def test_recommend_returns_prediction_envelope(client, monkeypatch):
    def recommend(request):
        assert request["target_problem_id"] == "problem-1"
        return {
            "model_name": "study-assistant-learning-model",
            "model_version": "4",
            "warnings": [],
            "data": {"p_correct": 0.8, "predicted_correct": True},
        }

    monkeypatch.setattr(api.hobby, "recommend", recommend, raising=False)
    response = client.post(
        "/api/v1/hobby-learning/recommend",
        json={
            "learner_id": "learner-1",
            "subject": "Mathematics",
            "topic": "ratios",
            "hobby": "Music",
            "target_problem_id": "problem-1",
        },
    )

    assert response.status_code == 200
    assert response.json()["model_version"] == "4"
    assert response.json()["data"]["p_correct"] == 0.8
    assert_request_id(response)


def test_recommend_is_unavailable_without_model_implementation(client):
    response = client.post(
        "/api/v1/hobby-learning/recommend",
        json={
            "learner_id": "learner-1",
            "subject": "Mathematics",
            "topic": "ratios",
            "hobby": "Music",
            "target_problem_id": "problem-1",
        },
    )

    assert response.status_code == 503
    assert "integration is unavailable" in response.json()["detail"]
    assert_request_id(response)


def test_invalid_recommendation_payload_returns_sanitized_422(client):
    response = client.post(
        "/api/v1/hobby-learning/recommend",
        json={"learner_id": "private learner", "subject": "Mathematics"},
    )

    assert response.status_code == 422
    assert "request_id" in response.json()
    assert "input" not in str(response.json())


def test_evaluate_flags_blank_and_low_confidence_answers(client, monkeypatch):
    def evaluate(answers, confidences):
        assert confidences == {"Q21": 0.2}
        return {
            "model_name": "study-assistant-answer-scoring-model",
            "model_version": "2",
            "warnings": [],
            "data": {
                "results": [
                    {"qid": "Q21", "estimated_mark": 1.0},
                    {"qid": "Q22", "estimated_mark": 0.0},
                ]
            },
        }

    monkeypatch.setattr(api.feedback, "evaluate", evaluate, raising=False)
    response = client.post(
        "/api/v1/answer-feedback/evaluate",
        json={
            "assessment_id": "assessment-1",
            "answers": {"Q21": "some answer words", "Q22": ""},
            "ocr_confidence": {"Q21": 0.2},
        },
    )

    assert response.status_code == 200
    results = response.json()["data"]["results"]
    assert all(result["human_review_required"] for result in results)
    assert_request_id(response)


def test_evaluate_rejects_out_of_range_ocr_confidence(client):
    response = client.post(
        "/api/v1/answer-feedback/evaluate",
        json={
            "assessment_id": "assessment-1",
            "answers": {"Q21": "a sufficiently long answer"},
            "ocr_confidence": {"Q21": 1.2},
        },
    )

    assert response.status_code == 422
    assert_request_id(response)


def test_evaluate_maps_unknown_question_value_error_to_422(client, monkeypatch, caplog):
    def evaluate(answers, confidences):
        raise ValueError(f"Unknown question: {next(iter(answers.values()))}")

    monkeypatch.setattr(api.feedback, "evaluate", evaluate, raising=False)
    response = client.post(
        "/api/v1/answer-feedback/evaluate",
        json={
            "assessment_id": "assessment-1",
            "answers": {"Q999": "private student answer"},
        },
    )

    assert response.status_code == 422
    assert "private student answer" not in response.text
    assert "private student answer" not in caplog.text
    assert_request_id(response)


def test_approve_returns_estimate_override_and_model_version(client):
    write_model_artifacts(api.MODEL_DIR, include_models=False)
    response = client.post(
        "/api/v1/answer-feedback/approve",
        json={
            "assessment_id": "assessment-1",
            "qid": "Q21",
            "estimated_mark": 1.0,
            "approved_mark": 2.0,
            "max_mark": 2.0,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "teacher_approved"
    assert response.json()["estimated_mark"] == 1.0
    assert response.json()["approved_mark"] == 2.0
    assert response.json()["model_version"] == "2"
    assert_request_id(response)


@pytest.mark.parametrize(
    ("approved_mark", "max_mark"),
    [(2.5, 2.0), (-0.5, 2.0), (1.0, 3.0)],
)
def test_approve_rejects_invalid_marks(client, approved_mark, max_mark):
    response = client.post(
        "/api/v1/answer-feedback/approve",
        json={
            "assessment_id": "assessment-1",
            "qid": "Q21",
            "estimated_mark": 1.0,
            "approved_mark": approved_mark,
            "max_mark": max_mark,
        },
    )

    assert response.status_code == 422
    assert_request_id(response)
