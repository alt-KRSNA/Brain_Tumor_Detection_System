from pathlib import Path
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
CLASS_FILE = BASE_DIR / "models" / "class_names.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
if not CLASS_FILE.exists():
    raise FileNotFoundError(f"class_names.json not found:\n{CLASS_FILE}")

with open(CLASS_FILE, "r") as f:
    CLASS_NAMES = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():
    """Load the trained model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# Load model once
model = load_model()


def preprocess_image(image_path: str):
    """Preprocess an input image."""

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found:\n{image_path}")

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(DEVICE)

    return image


def predict_image(image_path: str):
    """Predict the tumor class for an MRI image."""

    image = preprocess_image(image_path)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    prediction = CLASS_NAMES[predicted.item()]
    confidence_score = round(confidence.item() * 100, 2)

    probability_dict = {
        class_name: round(probabilities[0][i].item() * 100, 2)
        for i, class_name in enumerate(CLASS_NAMES)
    }

    return {
        "prediction": prediction,
        "confidence": confidence_score,
        "probabilities": probability_dict
    }


if __name__ == "__main__":

    image_path = input("Enter image path: ")

    result = predict_image(image_path)

    print(f"\nPrediction  : {result['prediction']}")
    print(f"Confidence : {result['confidence']} %")

    print("\nClass Probabilities")

    for class_name, probability in result["probabilities"].items():
        print(f"{class_name:<15}: {probability:.2f}%")