import io
import torch
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, field_validator
from PIL import Image

from models.classifier import MNISTClassifier

router = APIRouter()

CHECKPOINT_PATH = "checkpoints/best.pt"

model = MNISTClassifier()

if CHECKPOINT_PATH is not None:
    model.load(CHECKPOINT_PATH)

class DigitImage(BaseModel):
    pixels: list[float]

    @field_validator("pixels")
    @classmethod
    def validate_length(cls, v):
        if len(v) != 784:
            raise ValueError(f"Expected 784 pixel values (28x28 flattened), got {len(v)}")
        return v


def run_inference(x: torch.Tensor):
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        predicted = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0][predicted])
    return predicted, confidence


@router.get("/")
def home():
    return {"status": "ok", "message": "Welcome to the MNIST server"}


@router.get("/health")
def health():
    return {"status": "ok", "message": "Server is healthy"}


@router.post("/predict")
def predict(image: DigitImage):
    """Accepts a flat list of 784 pixel values, normalized 0-1."""
    try:
        x = torch.tensor(image.pixels, dtype=torch.float32).unsqueeze(0)
        predicted, confidence = run_inference(x)
        return {"predicted_digit": predicted, "confidence": round(confidence, 4)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    """Accepts an uploaded PNG/JPEG image of a handwritten digit."""
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use PNG or JPEG.",
        )

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("L")  # grayscale
        img = img.resize((28, 28))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read or process the image file")

    pixels = torch.tensor(list(img.getdata()), dtype=torch.float32) / 255.0
    x = pixels.unsqueeze(0)

    try:
        predicted, confidence = run_inference(x)
        return {"predicted_digit": predicted, "confidence": round(confidence, 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")