import json

from fastapi import APIRouter, status
from pathlib import Path 
from app.schemas import ReqBody, ModelInfo
from app.services import predict_message

router = APIRouter()

@router.post("/predict", status_code=status.HTTP_200_OK)
async def classify_message(req: ReqBody):
    message = req.message
    prediction, confidence = predict_message(message)
    confidence = float(confidence) * 100
    return {
        "message": message, 
        "prediction": prediction, 
        "confidence": confidence
        }

@router.get('/model-info', response_model=ModelInfo)
async def get_model_info():
    model_dir = Path(__file__).resolve().parents[1]
    metadata = model_dir / "model" / "metadata.json"

    with open(metadata, "r") as f:
        data = json.load(f)

    return data
"""
@router.get("/health")
async def get_model_health():
"""