import torch
import torch.nn as nn
import torch.optim as optim

from data.fashion_data import train_loader, val_loader
from models.cnn import FashionNN
from trainkit import Trainer

class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 30
    lr = 0.001
    patience = 5

def main():
    model = FashionNN()
    model.to(Config.device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_function,
        optimizer=optimizer,
        device=Config.device,
        checkpoint_dir="checkpoints",
        patience=Config.patience,
        wandb_project="trainkit-fashionmnist",
    )
    trainer.train(Config.epochs)

if __name__ == "__main__":
    main()