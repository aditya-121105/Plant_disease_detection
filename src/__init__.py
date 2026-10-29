"""
Plant Disease Classification using Pretrained ResNet-18
Author: Kumarkhaniya Aditya
"""

__version__ = "1.0.0"
__author__ = "Kumarkhaniya Aditya"

from .config import Config
from .model import PlantDiseaseModel
from .dataset import PlantVillageDataset
from .trainer import Trainer

__all__ = ["Config", "PlantDiseaseModel", "PlantVillageDataset", "Trainer"]
