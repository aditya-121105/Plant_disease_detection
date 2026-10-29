"""
Deep Learning CNN model based on Transfer Learning with ResNet-18.
"""

import torch
import torch.nn as nn
from torchvision import models


class PlantDiseaseModel(nn.Module):
    """
    Convolutional Neural Network for Plant Disease Classification based on ResNet-18.

    Architecture Overview:
    1. Convolutional Backbone (ResNet-18):
       - 18 layers deep with residual skip connections.
       - Extracts rich spatial hierarchies: edges, textures, lesions, and disease spots.
       - Pretrained on ImageNet-1k dataset.
    2. Batch Normalization:
       - Accelerates training and stabilizes weight updates.
    3. Residual Shortcuts:
       - Mitigates vanishing gradient degradation in deep architectures.
    4. Custom Classification Head:
       - Replaces standard 1000-class ImageNet head.
       - Adds Dropout for regularization and Linear projection to num_classes.
    """

    def __init__(self, num_classes=15, pretrained=True, freeze_backbone=True, dropout=0.2):
        super().__init__()
        self.num_classes = num_classes

        # Load ResNet-18 with pretrained weights (supporting both modern and legacy APIs)
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
        except AttributeError:
            self.model = models.resnet18(pretrained=pretrained)

        # Freeze feature extraction layers to preserve learned general visual filters
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace final fully connected classification layer
        in_features = self.model.fc.in_features
        if dropout > 0.0:
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """Forward pass through ResNet-18 backbone and classifier head."""
        return self.model(x)

    def unfreeze_backbone(self, num_blocks=None):
        """
        Unfreeze backbone parameters for fine-tuning.
        If num_blocks is specified (e.g., 1 or 2), unfreezes only the last residual blocks (layer4, layer3).
        """
        if num_blocks is None:
            for param in self.model.parameters():
                param.requires_grad = True
            print("Full backbone unfreezed for fine-tuning.")
        else:
            blocks = [self.model.layer4, self.model.layer3, self.model.layer2, self.model.layer1]
            for i in range(min(num_blocks, len(blocks))):
                for param in blocks[i].parameters():
                    param.requires_grad = True
            print(f"Unfreezed top {num_blocks} residual blocks for fine-tuning.")

    def count_parameters(self):
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
