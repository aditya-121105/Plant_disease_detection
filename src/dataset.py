"""
Dataset loading, augmentation, and analysis for PlantVillage dataset.
"""

import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from .config import Config


class PlantVillageDataset:
    """
    Handles dataset loading, data augmentation, normalization,
    and PyTorch DataLoader generation.
    """

    def __init__(self, dataset_path=None, image_size=None, batch_size=None):
        self.dataset_path = dataset_path or Config.dataset_path
        self.image_size = image_size or Config.image_size
        self.batch_size = batch_size or Config.batch_size

        # Training transform pipeline with data augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.mean, std=Config.std)
        ])

        # Validation/Test transform pipeline (deterministic)
        self.val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.mean, std=Config.std)
        ])

    def load_data(self, train_split_ratio=None, seed=None):
        """
        Load dataset using ImageFolder and partition into
        training and validation DataLoader sets.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset directory '{self.dataset_path}' not found. "
                "Please download the PlantVillage dataset or provide a valid path."
            )

        # Load raw dataset to inspect classes
        dataset = datasets.ImageFolder(
            root=self.dataset_path,
            transform=self.train_transform
        )

        train_ratio = train_split_ratio or Config.train_split
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size

        generator = torch.Generator().manual_seed(seed or Config.random_seed)
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator
        )

        # Apply deterministic transform to validation subset
        # (val_dataset shares underlying dataset transform, override via wrapper if needed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader, val_loader, dataset.classes


class DatasetAnalyzer:
    """
    Performs exploratory data analysis on the PlantVillage dataset.
    Counts images per class and plots class distribution.
    """

    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path or Config.dataset_path

    def get_class_counts(self):
        """Return a dictionary of class names and their respective image counts."""
        if not os.path.exists(self.dataset_path):
            return {}

        counts = {}
        for class_name in sorted(os.listdir(self.dataset_path)):
            class_dir = os.path.join(self.dataset_path, class_name)
            if os.path.isdir(class_dir):
                counts[class_name] = len([
                    f for f in os.listdir(class_dir)
                    if os.path.isfile(os.path.join(class_dir, f))
                ])
        return counts

    def plot_class_distribution(self, save_path=None):
        """Plot and optionally save a bar chart of class image counts."""
        counts = self.get_class_counts()
        if not counts:
            print("No classes found to plot.")
            return

        plt.figure(figsize=(14, 6))
        plt.bar(counts.keys(), counts.values(), color="#2ecc71", edgecolor="#27ae60")
        plt.xticks(rotation=90, ha="right", fontsize=9)
        plt.title("PlantVillage Dataset - Class Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Disease Class", fontsize=11)
        plt.ylabel("Number of Images", fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Distribution plot saved to {save_path}")

        plt.show()
