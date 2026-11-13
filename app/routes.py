
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from PIL import Image
import tempfile
import os

from .services import BrainTumorClassifier
from .schemas import QuestionRequest, QuestionResponse

from genai.rag_pipeline import ask_medical_question


router = APIRouter()

classifier = BrainTumorClassifier()


ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "gif"
}


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.filename is None:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded."
        )

    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Only jpg, jpeg, png, bmp and gif images are allowed."
        )

    suffix = Path(file.filename).suffix

    temp_path = None

    try:

        contents = await file.read()

        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(contents)
            temp_file.flush()

            temp_path = temp_file.name

        # Verify image
        try:

            image = Image.open(temp_path)
            image.verify()

        except Exception:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid image."
            )

        result = classifier.predict(temp_path)

        return result

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/health")
async def health():

    return classifier.health_check()


@router.get("/classes")
async def get_classes():

    return {
        "classes": classifier.get_class_names()
    }


@router.post(
    "/ask",
    response_model=QuestionResponse,
    tags=["medical assistant"]
)
async def ask_question(request: QuestionRequest):
    """
    Answer educational brain tumor questions using RAG.
    """

    try:

        result = ask_medical_question(
            question=request.question
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer. Please try again later."
        )
