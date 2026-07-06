import torch
import pandas as pd
from models.classifier import MNISTClassifier
from models.data import test_loader 

CHECKPOINT_PATH = "checkpoints/best.pt"

model = MNISTClassifier()
model = model.load(CHECKPOINT_PATH) 


def predict_all(loader):
    predictions = []
    with torch.no_grad():
        for X in loader: 
            logits = model(X)
            predicted = torch.argmax(logits, dim=1)
            predictions.extend(predicted.tolist())
    return predictions


if __name__ == "__main__":
    preds = predict_all(test_loader)
    print(f"Generated {len(preds)} predictions")
    print(f"First 10: {preds[:10]}")

    submission = pd.DataFrame({
        "ImageId": range(1, len(preds) + 1),
        "Label": preds,
    })
    submission.to_csv("./data/submission.csv", index=False)
    print("Saved submission.csv")