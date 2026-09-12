from pydantic import BaseModel 
from typing import Any


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
