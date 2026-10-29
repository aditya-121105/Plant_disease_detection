"""
Trainer class for model training, validation, and checkpointing.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from .config import Config


class Trainer:
    """
    Orchestrates the training, validation, metric recording,
    and checkpointing of the Plant Disease Model.
    """

    def __init__(self, model, train_loader, val_loader, criterion=None, optimizer=None, device=None, lr=None):
        self.device = device or Config.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion or nn.CrossEntropyLoss()
        self.learning_rate = lr or Config.learning_rate

        # Optimize only parameters with requires_grad=True
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optimizer or optim.Adam(trainable_params, lr=self.learning_rate)

        # Metric histories
        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []

        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self):
        """Train the model for one epoch over the training dataset."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        """Evaluate the model on the validation dataset."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / total
        val_acc = 100.0 * correct / total
        return val_loss, val_acc

    def train(self, epochs=None, save_best_path=None):
        """
        Execute full training loop across specified epochs.
        Tracks best validation accuracy and saves checkpoint.
        """
        num_epochs = epochs or Config.epochs
        save_path = save_best_path or Config.model_save_path

        print(f"Starting training on {self.device} for {num_epochs} epochs...\n")

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_acc.append(train_acc)
            self.val_acc.append(val_acc)

            is_best = val_acc > self.best_val_acc
            best_tag = ""
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                best_tag = " [Best Model Saved]"
                if save_path:
                    self.save_checkpoint(save_path, epoch=epoch, val_acc=val_acc)

            print(f"Epoch {epoch:02d}/{num_epochs:02d}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%{best_tag}")
            print("-" * 50)

        print(f"\nTraining completed! Best Validation Accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "best_val_acc": self.best_val_acc
        }

    def save_checkpoint(self, path, epoch=None, val_acc=None):
        """Save model state dictionary to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_acc": val_acc,
            "classes": getattr(self.model, "classes", None)
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load model weights from saved checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {path}")
