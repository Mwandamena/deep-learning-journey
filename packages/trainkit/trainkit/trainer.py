import torch
import torch.nn as nn
import os
import wandb

class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, device,
                 checkpoint_dir="checkpoints", patience=5, use_wandb=True, wandb_project="trainkit"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
        self.checkpoint_dir = checkpoint_dir
        self.best_val_acc = 0.0
        self.patience = patience
        self.epochs_without_improvement = 0
        self.use_wandb = use_wandb
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.use_wandb:
            wandb.init(project=wandb_project, config={
                "device": device,
                "patience": patience,
            })

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0

        for batch_index, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)

            with torch.autocast(device_type=self.device, enabled=(self.device == "cuda")):
                predictions = self.model(X)
                loss = self.loss_fn(predictions, y)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct, total, running_loss = 0, 0, 0.0

        for X, y in self.val_loader:
            X, y = X.to(self.device), y.to(self.device)
            predictions = self.model(X)
            loss = self.loss_fn(predictions, y)
            running_loss += loss.item()

            predicted_labels = predictions.argmax(dim=1)
            correct += (predicted_labels == y).sum().item()
            total += y.size(0)

        avg_loss = running_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def save_checkpoint(self, epoch, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict()
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def train(self, epochs):
        print(f"Starting training for up to {epochs} epochs on {self.device}...")
        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            val_loss, val_acc = self.evaluate()
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )

            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                })

            self.save_checkpoint(epoch + 1, "latest.pt")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch + 1, "best.pt")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping: no improvement for {self.patience} epochs.")
                    break

        print(f"Training complete! Best val_acc: {self.best_val_acc:.4f}")
        if self.use_wandb:
            wandb.finish()