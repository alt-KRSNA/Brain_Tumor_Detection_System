# Brain Tumor Classification - Application Guide

## Overview

This application provides both a **FastAPI backend** and a **Streamlit frontend** for brain tumor classification using deep learning.

### Components

1. **FastAPI Backend** - RESTful API for predictions
2. **Streamlit Frontend** - Interactive web interface
3. **ML Model** - ResNet18 trained on brain MRI images

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the model file exists at:
```
models/best_model.pth
```

## Running the Application

### Option 1: Using Streamlit (Recommended for UI)

Run the Streamlit frontend directly:

```bash
streamlit run app/streamlit_app.py
```

This will open an interactive web interface in your browser at `http://localhost:8501`

**Features:**
- Upload or select example images
- View real-time predictions
- See confidence scores and probability distributions
- Browse tumor classification information
- Download results as text

### Option 2: Using FastAPI Backend

Start the FastAPI server:

```bash
# From the project root directory
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or run directly:

```bash
python app/main.py
```

**API Endpoints:**

- `GET /` - Root endpoint with API info
- `GET /api/health` - Health check
- `GET /api/classes` - Get available tumor classes
- `POST /api/predict` - Predict tumor from image (multipart file upload)
- `GET /api/info` - API information
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

**Example API Request (Python):**

```python
import requests
from pathlib import Path

# Upload image for prediction
image_path = Path("dataset/Testing/glioma/image.jpg")

with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/api/predict", files=files)
    
print(response.json())
# Output:
# {
#     "prediction": "glioma",
#     "confidence": 95.23,
#     "class_names": ["glioma", "meningioma", "notumor", "pituitary"]
# }
```

### Option 3: Combined Setup (Backend + Frontend)

1. Start the backend in one terminal:
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. Start the frontend in another terminal:
```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
Medical Image Analysis/
├── app/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # FastAPI app setup
│   ├── routes.py            # API endpoints
│   ├── schemas.py           # Pydantic models
│   ├── services.py          # Business logic
│   ├── streamlit_app.py     # Streamlit frontend
│   └── README.md            # This file
├── models/
│   └── best_model.pth       # Pre-trained model
├── dataset/
│   ├── Training/            # Training images
│   └── Testing/             # Test images
├── src/
│   ├── train.py             # Training script
│   ├── inference.py         # Original inference code
│   ├── dataset.py           # Dataset utilities
│   └── test_torch.py        # Testing utilities
├── notebooks/
│   └── data_exploration.ipynb
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Tumor Classification Classes

1. **Glioma** - Tumor starting in glial cells
2. **Meningioma** - Tumor in the meninges (membrane around brain)
3. **No Tumor** - No tumor detected
4. **Pituitary** - Tumor in the pituitary gland

## Key Features

### Streamlit App Features
- 📤 Image upload and preview
- 🔍 Real-time predictions with confidence scores
- 📊 Interactive probability charts
- 📈 Confidence gauge visualization
- 🏥 Detailed tumor information
- 📚 Browse example images from dataset
- 💾 Download results
- ⚙️ System information display

### FastAPI Features
- RESTful API design
- CORS support for cross-origin requests
- Automatic API documentation (Swagger UI)
- File upload handling with validation
- Error handling and validation
- Health check endpoint
- Asynchronous processing
- Temporary file cleanup

## Configuration

### System Requirements

- **Python**: 3.8+
- **GPU** (optional): NVIDIA GPU with CUDA for faster inference
- **RAM**: At least 4GB recommended
- **Storage**: ~200MB for model and dependencies

### Device Selection

The application automatically:
- Uses CUDA GPU if available
- Falls back to CPU if GPU is not detected

### Model Configuration

Model specifications are defined in:
- `app/services.py` - BrainTumorClassifier class
- Input size: 224x224 pixels
- Architecture: ResNet18
- Output classes: 4 tumor types

## Troubleshooting

### Model Not Found
- Ensure `models/best_model.pth` exists
- Check the model path in `app/services.py`

### Port Already in Use
```bash
# Change port for FastAPI
python -m uvicorn app.main:app --port 8001

# Change port for Streamlit
streamlit run app/streamlit_app.py --server.port 8502
```

### GPU Not Detected
- Install CUDA and cuDNN
- Verify PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`

### Image Upload Issues
- Ensure image is in supported format: JPG, PNG, BMP, GIF
- Check image file size and quality
- Verify image contains brain MRI scan

## Performance Tips

1. **GPU Usage**: Use GPU for faster predictions (~50-100x faster)
2. **Batch Processing**: For API, send multiple requests concurrently
3. **Caching**: Streamlit caches the model in memory for faster subsequent predictions
4. **Image Preprocessing**: Ensure images are properly formatted (224x224 or larger)

## API Response Format

### Success Response
```json
{
    "prediction": "glioma",
    "confidence": 95.23,
    "class_names": ["glioma", "meningioma", "notumor", "pituitary"]
}
```

### Health Check Response
```json
{
    "status": "healthy",
    "model": "Brain Tumor Classifier",
    "device": "cuda",
    "classes": ["glioma", "meningioma", "notumor", "pituitary"]
}
```

## Disclaimer

⚠️ **IMPORTANT**: This application is designed for **educational and research purposes only**. It should not be used for clinical decision-making without professional medical review. Always consult with qualified healthcare professionals before making any medical decisions based on AI predictions.

## Future Enhancements

- [ ] Batch prediction processing
- [ ] Model performance metrics dashboard
- [ ] User authentication
- [ ] Database storage for predictions
- [ ] Model versioning
- [ ] Real-time model updates
- [ ] Multi-model ensemble
- [ ] Export predictions to DICOM format
- [ ] Integration with medical imaging software
- [ ] Advanced visualization tools

## Support & Contact

For questions or issues, please refer to the project documentation or contact the development team.

---

**Version**: 1.0.0  
**Last Updated**: 2026  
**Status**: Production Ready
