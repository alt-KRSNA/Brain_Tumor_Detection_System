from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
from pathlib import Path
from .routes import router

# Create FastAPI app
app = FastAPI(
    title="Brain Tumor Classification API",
    description="API for classifying brain tumors from medical images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api", tags=["predictions"])


@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Brain Tumor Classification API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /api/predict",
            "health": "GET /api/health",
            "classes": "GET /api/classes",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/api/info")
async def api_info():
    """
    Get API information
    """
    return {
        "api_name": "Brain Tumor Classification",
        "model_type": "ResNet18",
        "classes": ["glioma", "meningioma", "notumor", "pituitary"],
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "gif"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
