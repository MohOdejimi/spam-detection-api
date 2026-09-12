import json

from fastapi import APIRouter, status, HTTPException
from pathlib import Path 
from app.schemas import ReqBody, ModelInfo, HealthInfo
from app.services import predict_message
from app.services import model, vectorizer, metadata

router = APIRouter()

@router.post("/predict", status_code=status.HTTP_200_OK)
async def classify_message(req: ReqBody):
    message = req.message
    if message is "":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Unprocessable Content"
        )
    prediction, confidence = predict_message(message)
    confidence = float(confidence) * 100
    return {
        "message": message, 
        "prediction": prediction, 
        "confidence": confidence
        }

@router.get('/model-info', response_model=ModelInfo)
async def get_model_info():
    return metadata

def is_model_healthy(model, vectorizer):
    if vectorizer is None or model is None:
        return False 
    try:
        predict_message('text')
        return True 
    except Exception:
        return False
    

@router.get("/health", response_model=HealthInfo)
async def get_model_health():
    model_health_status = is_model_healthy(model, vectorizer)
    if not model_health_status:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "model_loaded": False, 
                "model_algorithm": None
            }
        )
    else:
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_algorithm": metadata['algorithm'],
        }
    