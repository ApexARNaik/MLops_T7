import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]
AnswerText = Annotated[str, StringConstraints(max_length=10_000)]


class HobbyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learner_id: NonEmptyString
    subject: NonEmptyString
    topic: NonEmptyString
    hobby: NonEmptyString
    target_problem_id: NonEmptyString


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment_id: NonEmptyString
    answers: dict[NonEmptyString, AnswerText]
    ocr_confidence: dict[NonEmptyString, float] | None = None

    @field_validator("ocr_confidence")
    @classmethod
    def validate_ocr_confidence(
        cls, values: dict[str, float] | None
    ) -> dict[str, float] | None:
        if values is not None and any(
            not math.isfinite(confidence) or not 0 <= confidence <= 1
            for confidence in values.values()
        ):
            raise ValueError("OCR confidence values must be between 0 and 1.")
        return values


class ApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment_id: NonEmptyString
    qid: NonEmptyString
    estimated_mark: float = Field(ge=0, le=2)
    approved_mark: float = Field(ge=0, le=2)
    max_mark: float = Field(default=2, gt=0, le=2)

    @model_validator(mode="after")
    def validate_marks(self) -> "ApproveRequest":
        if not all(
            math.isfinite(mark)
            for mark in (self.estimated_mark, self.approved_mark, self.max_mark)
        ):
            raise ValueError("Marks must be finite numbers.")
        if self.estimated_mark > self.max_mark:
            raise ValueError("estimated_mark cannot exceed max_mark.")
        if self.approved_mark > self.max_mark:
            raise ValueError("approved_mark cannot exceed max_mark.")
        return self
