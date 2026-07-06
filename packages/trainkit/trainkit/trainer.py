import os
import torch
import wandb


class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, device,
                 checkpoint_dir="checkpoints", patience=5, min_delta=0.001,
                 monitor="val_loss", use_wandb=True, wandb_project="trainkit",
                 resume_from=None):
   
        assert monitor in ("val_loss", "val_acc") 
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.epochs_without_improvement = 0
        self.use_wandb = use_wandb
        self.start_epoch = 0

        self.best_metric = float("inf") if monitor == "val_loss" else 0.0

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if resume_from is not None:
            self._load_checkpoint(resume_from)

        if self.use_wandb:
            wandb.init(project=wandb_project, config={
                "device": device,
                "patience": patience,
                "min_delta": min_delta,
                "monitor": monitor,
                "resumed_from": resume_from,
            })

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])

        self.start_epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("best_metric", self.best_metric)

        print(f"Resumed from {path}: epoch={self.start_epoch}, best_{self.monitor}={self.best_metric:.4f}")

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
            'scaler_state': self.scaler.state_dict(),
            'best_metric': self.best_metric,
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def _is_improvement(self, current):
        if self.monitor == "val_loss":
            return current < (self.best_metric - self.min_delta)
        else:
            return current > (self.best_metric + self.min_delta)

    def train(self, epochs):
        remaining = epochs - self.start_epoch
        if remaining <= 0:
            print(f"Nothing to do: already at epoch {self.start_epoch}, target was {epochs}.")
            return

        print(f"Starting training from epoch {self.start_epoch + 1} to {epochs} on {self.device}...")

        for epoch in range(self.start_epoch, epochs):
            train_loss = self.train_one_epoch()
            val_loss, val_acc = self.evaluate()
            current_metric = val_loss if self.monitor == "val_loss" else val_acc

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

            if self._is_improvement(current_metric):
                self.best_metric = current_metric
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch + 1, "best.pt")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(
                        f"Early stopping: no {self.monitor} improvement "
                        f"(min_delta={self.min_delta}) for {self.patience} epochs."
                    )
                    break

        print(f"Training complete! Best {self.monitor}: {self.best_metric:.4f}")
        if self.use_wandb:
            wandb.finish()