from pydantic import BaseModel
from typing import Optional


class PredictionResponse(BaseModel):
    """Response model for image prediction"""
    prediction: str
    confidence: float
    class_names: list[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "glioma",
                "confidence": 95.23,
                "class_names": ["glioma", "meningioma", "notumor", "pituitary"]
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
