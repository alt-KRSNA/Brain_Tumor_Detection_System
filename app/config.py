"""Application Configuration"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
APP_DIR = PROJECT_ROOT / "app"
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_DIR = PROJECT_ROOT / "dataset"

# Model configuration
MODEL_PATH = MODELS_DIR / "best_model.pth"
MODEL_ARCHITECTURE = "resnet18"
MODEL_INPUT_SIZE = (224, 224)

# Classification classes
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# API configuration
API_TITLE = "Brain Tumor Classification API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "API for classifying brain tumors from medical images"
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True
API_WORKERS = 1

# Streamlit configuration
STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# Device configuration
USE_GPU = True  # Set to False to force CPU usage

# File upload configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif"}

# CORS configuration
CORS_ORIGINS = ["*"]
CORS_CREDENTIALS = True
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Prediction confidence threshold for warnings
CONFIDENCE_WARNING_THRESHOLD = 75  # Show warning if confidence < 75%
CONFIDENCE_ALERT_THRESHOLD = 50   # Show alert if confidence < 50%
