"""
Utility functions for metrics visualization, confusion matrix, and single-image prediction.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from .config import Config

# Standard 15-class PlantVillage dataset label mapping
DEFAULT_PLANT_CLASSES = [
    "Pepper bell - Bacterial spot",
    "Pepper bell - Healthy",
    "Potato - Early blight",
    "Potato - Late blight",
    "Potato - Healthy",
    "Tomato - Bacterial spot",
    "Tomato - Early blight",
    "Tomato - Late blight",
    "Tomato - Leaf mold",
    "Tomato - Septoria leaf spot",
    "Tomato - Spider mites (Two-spotted spider mite)",
    "Tomato - Target spot",
    "Tomato - Yellow leaf curl virus",
    "Tomato - Mosaic virus",
    "Tomato - Healthy"
]


def plot_training_graphs(trainer, save_dir=None):
    """
    Plot Training vs Validation Loss and Accuracy curves.
    Optionally saves the plots to save_dir.
    """
    epochs = range(1, len(trainer.train_losses) + 1)
    save_dir = save_dir or Config.assets_dir
    os.makedirs(save_dir, exist_ok=True)

    # 1. Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, trainer.train_losses, label="Train Loss", color="#e74c3c", marker="o", linewidth=2)
    plt.plot(epochs, trainer.val_losses, label="Validation Loss", color="#3498db", marker="s", linewidth=2)
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("CrossEntropy Loss", fontsize=11)
    plt.title("Training vs Validation Loss", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    loss_path = os.path.join(save_dir, "training_loss.png")
    plt.savefig(loss_path, dpi=300)
    print(f"Loss curve saved to: {loss_path}")
    plt.show()

    # 2. Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, trainer.train_acc, label="Train Accuracy", color="#2ecc71", marker="o", linewidth=2)
    plt.plot(epochs, trainer.val_acc, label="Validation Accuracy", color="#9b59b6", marker="s", linewidth=2)
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.title("Training vs Validation Accuracy", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    acc_path = os.path.join(save_dir, "training_accuracy.png")
    plt.savefig(acc_path, dpi=300)
    print(f"Accuracy curve saved to: {acc_path}")
    plt.show()


def evaluate_model(model, val_loader, device=None, class_names=None, save_dir=None):
    """
    Run full evaluation on validation loader.
    Prints classification report and renders confusion matrix.
    """
    device = device or Config.device
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    classes = class_names or [f"Class {i}" for i in range(len(set(all_labels)))]

    print("\n" + "=" * 60)
    print("                CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(all_labels, all_preds, target_names=classes[:len(set(all_labels))], digits=4)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix - Plant Disease Detection", fontsize=14, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cm_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        print(f"Confusion matrix saved to: {cm_path}")

    plt.show()
    return report, cm


def predict_single_image(model, image_path, device=None, class_names=None, top_k=3):
    """
    Predict plant disease for a single input leaf image.
    Returns list of tuples: (class_name, confidence_percentage)
    """
    device = device or Config.device
    classes = class_names or DEFAULT_PLANT_CLASSES

    transform = transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.mean, std=Config.std)
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(classes)))
    results = []
    for prob, idx in zip(top_probs, top_indices):
        idx_val = idx.item()
        cname = classes[idx_val] if idx_val < len(classes) else f"Class {idx_val}"
        results.append((cname, prob.item() * 100.0))

    return results
