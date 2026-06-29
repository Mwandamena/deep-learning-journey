import torch
from fastapi import APIRouter
from pydantic import BaseModel

from models.classifier import MNISTClassifier

router = APIRouter()

model = MNISTClassifier()

class DigitImage(BaseModel):
    pixels: list[float] 


@router.get("/")
def home():
    return {"status": "ok", "message": "Welcome to the MNIST server"}

@router.get("/health")
def health():
    return {"status": "ok", "message": "Server is healthy"}


@router.post("/predict")
def predict(image: DigitImage):
    x = torch.tensor(image.pixels, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        predicted = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0][predicted])

    return {"predicted_digit": predicted, "confidence": round(confidence, 4)}