#!/usr/bin/env python3
"""
CLI script to train ResNet-18 for Plant Disease Detection.

Usage:
    python train.py --data_dir PlantVillage --epochs 5 --batch_size 32 --lr 0.001
"""

import argparse
import os
from src.config import Config
from src.dataset import PlantVillageDataset
from src.model import PlantDiseaseModel
from src.trainer import Trainer
from src.utils import plot_training_graphs


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 for Plant Disease Detection")
    parser.add_argument("--data_dir", type=str, default=Config.dataset_path, help="Path to PlantVillage dataset folder")
    parser.add_argument("--epochs", type=int, default=Config.epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=Config.batch_size, help="Batch size for DataLoaders")
    parser.add_argument("--lr", type=float, default=Config.learning_rate, help="Learning rate for Adam optimizer")
    parser.add_argument("--save_path", type=str, default=Config.model_save_path, help="Path to save best model checkpoint")
    parser.add_argument("--unfreeze_blocks", type=int, default=0, help="Number of ResNet residual blocks to fine-tune (0 = fully frozen backbone)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Update config overrides
    Config.dataset_path = args.data_dir
    Config.epochs = args.epochs
    Config.batch_size = args.batch_size
    Config.learning_rate = args.lr
    Config.model_save_path = args.save_path

    Config.display()

    print(f"\n1. Loading dataset from '{Config.dataset_path}'...")
    dataset_loader = PlantVillageDataset(
        dataset_path=Config.dataset_path,
        image_size=Config.image_size,
        batch_size=Config.batch_size
    )

    train_loader, val_loader, classes = dataset_loader.load_data()
    print(f"   Found {len(classes)} classes: {classes[:5]} ... (+{len(classes)-5} more)")

    print("\n2. Initializing ResNet-18 Transfer Learning model...")
    model = PlantDiseaseModel(
        num_classes=len(classes),
        pretrained=Config.pretrained,
        freeze_backbone=Config.freeze_backbone,
        dropout=Config.dropout
    )
    model.classes = classes

    if args.unfreeze_blocks > 0:
        model.unfreeze_backbone(num_blocks=args.unfreeze_blocks)

    params = model.count_parameters()
    print(f"   Total parameters:     {params['total']:,}")
    print(f"   Trainable parameters: {params['trainable']:,}")
    print(f"   Frozen parameters:    {params['frozen']:,}")

    print("\n3. Training model...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=Config.learning_rate,
        device=Config.device
    )

    trainer.train(epochs=Config.epochs, save_best_path=Config.model_save_path)

    print("\n4. Generating and saving training curves...")
    plot_training_graphs(trainer, save_dir=Config.assets_dir)
    print("\nTraining workflow completed successfully!")


if __name__ == "__main__":
    main()
