
from pydantic import BaseModel, Field
from typing import Optional


class PredictionResponse(BaseModel):
    """Response model for image prediction."""

    prediction: str
    confidence: float
    class_names: list[str]

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "glioma",
                "confidence": 95.23,
                "class_names": [
                    "glioma",
                    "meningioma",
                    "notumor",
                    "pituitary"
                ]
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None


class QuestionRequest(BaseModel):
    """Request model for medical questions."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Educational brain tumor-related question"
    )


class SourceReference(BaseModel):
    """Reference to retrieved medical information."""

    filename: str
    page: int


class QuestionResponse(BaseModel):
    """Response model for RAG chatbot."""

    answer: str
    sources: list[SourceReference]
