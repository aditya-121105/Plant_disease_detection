"""
Configuration module for Plant Disease Detection.
Stores all hyperparameters, model settings, and paths.
"""

import os
import torch


class Config:
    """Hyperparameters and runtime settings for training and inference."""

    # Paths
    dataset_path = "PlantVillage"
    model_save_dir = "models"
    model_save_path = os.path.join(model_save_dir, "plant_disease_resnet18.pth")
    assets_dir = "assets"

    # Image & Preprocessing
    image_size = 224
    num_channels = 3
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training Hyperparameters
    batch_size = 32
    epochs = 5
    learning_rate = 0.001
    train_split = 0.8
    num_workers = 2
    random_seed = 42

    # Architecture
    backbone = "resnet18"
    pretrained = True
    freeze_backbone = True
    dropout = 0.2

    # Hardware Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def display(cls):
        """Print current configuration."""
        print("=" * 50)
        print(" Plant Disease Detection - Configuration")
        print("=" * 50)
        print(f" Dataset Path:      {cls.dataset_path}")
        print(f" Image Size:        {cls.image_size}x{cls.image_size}")
        print(f" Batch Size:        {cls.batch_size}")
        print(f" Epochs:            {cls.epochs}")
        print(f" Learning Rate:     {cls.learning_rate}")
        print(f" Backbone:          {cls.backbone} (pretrained={cls.pretrained})")
        print(f" Device:            {cls.device}")
        print("=" * 50)
