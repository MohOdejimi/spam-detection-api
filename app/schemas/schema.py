from typing import Any

from pydantic import BaseModel


class ReqBody(BaseModel):
    message: str


class ModelInfo(BaseModel):
    algorithm: str
    vectorizer: str
    vocabulary_size: float
    accuracy: float
    classification_report: Any
    train_samples: int
    test_samples: int


class HealthInfo(BaseModel):
    status: str
    model_loaded: bool
    model_algorithm: str
