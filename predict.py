#!/usr/bin/env python3
"""
CLI script to classify plant disease from a single leaf photograph.

Usage:
    python predict.py --image path/to/leaf.jpg [--model models/plant_disease_resnet18.pth]
"""

import argparse
import os
import torch
from src.config import Config
from src.model import PlantDiseaseModel
from src.utils import DEFAULT_PLANT_CLASSES, predict_single_image


def parse_args():
    parser = argparse.ArgumentParser(description="Predict plant disease from leaf image")
    parser.add_argument("--image", type=str, required=True, help="Path to input leaf image (JPG/PNG)")
    parser.add_argument("--model", type=str, default=Config.model_save_path, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to display")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
        return

    classes = DEFAULT_PLANT_CLASSES
    model = PlantDiseaseModel(num_classes=len(classes), pretrained=False, freeze_backbone=False)

    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=Config.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if checkpoint.get("classes"):
                classes = checkpoint["classes"]
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model checkpoint from: {args.model}")
    else:
        print(f"Note: Model checkpoint '{args.model}' not found.")
        print("Using ResNet-18 feature representations for demonstration.")

    model = model.to(Config.device)

    print(f"\nAnalyzing image: {args.image}...")
    results = predict_single_image(
        model=model,
        image_path=args.image,
        device=Config.device,
        class_names=classes,
        top_k=args.top_k
    )

    print("\n" + "=" * 55)
    print("        PLANT DISEASE DIAGNOSIS RESULTS")
    print("=" * 55)
    for rank, (cname, conf) in enumerate(results, 1):
        status_indicator = "🌱 [HEALTHY]" if "healthy" in cname.lower() else "⚠️  [DISEASE DETECTED]"
        print(f" {rank}. {cname:<40} {conf:6.2f}%  {status_indicator}")
    print("=" * 55)

    top_disease, top_conf = results[0]
    if "healthy" in top_disease.lower():
        print("\n✅ Diagnosis: Plant leaf appears HEALTHY.")
    else:
        print(f"\n⚠️  Diagnosis: High likelihood of {top_disease} ({top_conf:.1f}% confidence).")
        print("💡 Recommendation: Inspect surrounding foliage, isolate affected leaves, and consult agricultural guidelines.")


if __name__ == "__main__":
    main()
