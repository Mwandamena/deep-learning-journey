import io
import torch
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, field_validator
from PIL import Image

from models.cnn import FashionNN

router = APIRouter()

CHECKPOINT_PATH = "checkpoints/best.pt"

model = FashionNN()
model.load(CHECKPOINT_PATH)

FASHION_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class FashionImage(BaseModel):
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
    return {"status": "ok", "message": "Welcome to the FashionMNIST server"}


@router.get("/health")
def health():
    return {"status": "ok", "message": "Server is healthy"}


@router.post("/predict")
def predict(image: FashionImage):
    x = torch.tensor(image.pixels, dtype=torch.float32).reshape(1, 1, 28, 28)
    predicted, confidence = run_inference(x)
    return {
        "predicted_class": FASHION_LABELS[predicted],
        "confidence": round(confidence, 4),
    }


@router.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read or process the image file")

    pixels = torch.tensor(list(img.getdata()), dtype=torch.float32) / 255.0
    x = pixels.reshape(1, 1, 28, 28)

    predicted, confidence = run_inference(x)
    return {
        "predicted_class": FASHION_LABELS[predicted],
        "confidence": round(confidence, 4),
    }