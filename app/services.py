from pathlib import Path
from typing import Any, Dict
import sys

# Add project root to Python path
BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.inference import predict_image


class BrainTumorClassifier:
    """Service for brain tumor classification."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path

    def predict(self, image_path: str) -> Dict[str, Any]:
        """Predict the tumor class for an image."""

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        result = predict_image(str(image_path))

        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "class_probabilities": result["probabilities"]
        }

    def get_class_names(self):
        """Return available class names."""

        from src.inference import CLASS_NAMES
        return CLASS_NAMES

    def health_check(self):
        """Return service status."""

        return {
            "status": "healthy",
            "model_loaded": True
        }