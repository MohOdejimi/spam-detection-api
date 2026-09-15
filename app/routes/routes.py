from fastapi import APIRouter, HTTPException, status

from app.schemas import HealthInfo, ModelInfo, ReqBody
from app.services import metadata, model, predict_message, vectorizer

router = APIRouter()


@router.post("/predict", status_code=status.HTTP_200_OK)
async def classify_message(req: ReqBody):
    message = req.message
    if not message:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Unprocessable Content",
        )
    prediction, confidence = predict_message(message)
    confidence = float(confidence) * 100
    return {"message": message, "prediction": prediction, "confidence": confidence}


@router.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    return metadata


def is_model_healthy(model, vectorizer):
    if vectorizer is None or model is None:
        return False
    try:
        predict_message("text")
        return True
    except Exception:  # noqa: BLE001 -- health check needs "anything failed", not a specific type
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
                "model_algorithm": None,
            },
        )
    else:
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_algorithm": metadata["algorithm"],
        }
