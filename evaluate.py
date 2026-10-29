#!/usr/bin/env python3
"""
CLI script to evaluate a saved Plant Disease model on the validation dataset.

Usage:
    python evaluate.py --data_dir PlantVillage --model_path models/plant_disease_resnet18.pth
"""

import argparse
import os
import torch
from src.config import Config
from src.dataset import PlantVillageDataset
from src.model import PlantDiseaseModel
from src.utils import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Plant Disease Detection model")
    parser.add_argument("--data_dir", type=str, default=Config.dataset_path, help="Path to PlantVillage dataset folder")
    parser.add_argument("--model_path", type=str, default=Config.model_save_path, help="Path to trained PyTorch checkpoint (.pth)")
    parser.add_argument("--batch_size", type=int, default=Config.batch_size, help="Batch size for validation loader")
    parser.add_argument("--assets_dir", type=str, default=Config.assets_dir, help="Directory to save evaluation plots")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading dataset from '{args.data_dir}'...")
    dataset_loader = PlantVillageDataset(
        dataset_path=args.data_dir,
        image_size=Config.image_size,
        batch_size=args.batch_size
    )

    _, val_loader, classes = dataset_loader.load_data()
    print(f"Validation dataset loaded. Classes: {len(classes)}")

    print(f"Loading model checkpoint from '{args.model_path}'...")
    model = PlantDiseaseModel(num_classes=len(classes), pretrained=False, freeze_backbone=False)

    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=Config.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            val_acc = checkpoint.get("val_acc")
            if val_acc:
                print(f"Checkpoint was saved with validation accuracy: {val_acc:.2f}%")
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint weights loaded successfully.")
    else:
        print(f"Warning: Checkpoint '{args.model_path}' not found. Evaluating with initialized weights for demonstration.")

    model = model.to(Config.device)
    evaluate_model(
        model=model,
        val_loader=val_loader,
        device=Config.device,
        class_names=classes,
        save_dir=args.assets_dir
    )


if __name__ == "__main__":
    main()
