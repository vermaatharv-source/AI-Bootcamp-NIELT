"""
Inference & Defect Detection Pipeline with Grad-CAM Explainable AI.
Can evaluate individual images or batch directories and output visual diagnostic cards.
"""

import argparse
import json
import os
import sys
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.cnn_model import build_model
from models.gradcam import GradCAM
from models.quality_grader import QualityGrader, QualityReport
from utils.transforms import preprocess_image_for_model
from utils.visualizer import generate_prediction_card


def load_trained_model(
    checkpoint_path: str = "checkpoints/best_model.pth",
    device: Optional[torch.device] = None
):
    """Loads trained model weights and class labels from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' not found. Please train the model first using train.py."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "custom_cnn")
    num_classes = checkpoint.get("num_classes", 8)
    class_names = checkpoint.get("class_names", [])

    model = build_model(num_classes=num_classes, model_type=model_type)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, device


def predict_single_image(
    image_path: str,
    model: torch.nn.Module,
    class_names: list,
    device: torch.device,
    image_size: int = 128,
    save_viz_path: Optional[str] = None
) -> QualityReport:
    """
    Performs full CNN inference, Grad-CAM attention calculation,
    and quality grading evaluation on a single image.
    """
    grader = QualityGrader(class_names=class_names)
    gradcam = GradCAM(model)

    pil_img = Image.open(image_path).convert("RGB")

    # Preprocess
    tensor, original_rgb = preprocess_image_for_model(
        pil_img,
        image_size=(image_size, image_size),
        device=device
    )

    # Model inference
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    # Generate Grad-CAM heatmap
    heatmap_2d, _ = gradcam.generate_heatmap(tensor)
    gradcam.remove_hooks()

    # Blend Grad-CAM heatmap with original RGB image
    heatmap_overlay_rgb = gradcam.overlay_heatmap(original_rgb, heatmap_2d, alpha=0.55)

    # Compute comprehensive quality grade report
    report = grader.grade(
        probabilities=probs,
        heatmap_2d=heatmap_2d,
        original_rgb=original_rgb,
        pil_image=pil_img
    )

    # Optional: Save visual side-by-side diagnostic card
    if save_viz_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_viz_path)), exist_ok=True)
        generate_prediction_card(original_rgb, heatmap_overlay_rgb, report, save_viz_path)

    return report


def print_report_card(report: QualityReport, image_path: str):
    """Prints a beautiful CLI summary card for the prediction."""
    print("=" * 64)
    print(f" FRUIT & VEGETABLE QUALITY GRADING REPORT")
    print("=" * 64)
    print(f" Image:                {os.path.basename(image_path)}")
    print(f" Detected Commodity:   {report.commodity}")
    print(f" Surface Condition:    {report.condition}")
    print(f" Assigned Grade:       {report.grade} ({report.grade_description})")
    print(f" Freshness Index:      {report.freshness_score}%")
    print(f" Defect Coverage:      {report.defect_percentage}%")
    print(f" Model Confidence:     {report.confidence * 100:.2f}%")
    print(f" Estimated Shelf-Life: {report.estimated_shelf_life}")
    print(f" Recommended Action:   {report.recommended_action}")
    print("-" * 64)
    print(" Class Probability Distribution:")
    for cls_name, prob in sorted(report.probabilities.items(), key=lambda x: x[1], reverse=True)[:4]:
        bar = "#" * int(prob * 30)
        print(f"   {cls_name:<16}: {prob * 100:5.1f}% | {bar}")
    print("=" * 64)


def parse_args():
    parser = argparse.ArgumentParser(description="Fruit & Vegetable Quality Prediction")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image file")
    parser.add_argument("--input-dir", type=str, default=None, help="Path to directory of test images")
    parser.add_argument("--output-dir", type=str, default="predictions", help="Directory to save diagnostic outputs")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model weights path")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image and not args.input_dir:
        # Default fallback to samples directory if exists
        if os.path.exists("data/samples"):
            args.input_dir = "data/samples"
        else:
            print("Please specify --image <path> or --input-dir <dir>.")
            sys.exit(1)

    model, class_names, device = load_trained_model(args.checkpoint)

    if args.image:
        viz_path = os.path.join(args.output_dir, f"diag_{os.path.basename(args.image)}")
        report = predict_single_image(args.image, model, class_names, device, save_viz_path=viz_path)
        print_report_card(report, args.image)
        print(f"Diagnostic image saved to: {viz_path}")

    elif args.input_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        images = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.lower().endswith(valid_extensions)
        ]
        if not images:
            print(f"No valid image files found in '{args.input_dir}'.")
            return

        print(f"Found {len(images)} images in '{args.input_dir}'. Running inference...\n")
        results = []
        for img_path in images:
            viz_path = os.path.join(args.output_dir, f"diag_{os.path.basename(img_path)}")
            report = predict_single_image(img_path, model, class_names, device, save_viz_path=viz_path)
            print(f"[{report.commodity}] {os.path.basename(img_path):<24} -> {report.grade} ({report.condition}) | Freshness: {report.freshness_score:5.1f}%")
            results.append({
                "file": os.path.basename(img_path),
                **report.to_dict()
            })

        # Save batch summary JSON
        summary_path = os.path.join(args.output_dir, "batch_grading_results.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nAll diagnostic cards and summary saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
