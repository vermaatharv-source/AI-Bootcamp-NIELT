"""
Evaluation script to compute test metrics, class-level precision/recall/F1,
confusion matrix, and diagnostic error analysis.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from models.cnn_model import build_model
from utils.transforms import get_val_transforms
from utils.visualizer import plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CNN Quality Grading Model")
    parser.add_argument("--test-dir", type=str, default="data/dataset/val", help="Path to test/val data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Checkpoint path")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for reports")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Model checkpoint '{args.checkpoint}' not found.")
        sys.exit(1)

    if not os.path.exists(args.test_dir):
        print(f"Error: Evaluation directory '{args.test_dir}' not found.")
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "custom_cnn")
    class_names = checkpoint.get("class_names", [])
    num_classes = len(class_names)
    image_size = checkpoint.get("image_size", 128)

    model = build_model(num_classes=num_classes, model_type=model_type)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_transforms = get_val_transforms((image_size, image_size))
    dataset = ImageFolder(args.test_dir, transform=val_transforms)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"============================================================")
    print(f" Fruit & Vegetable Quality Grading CNN - Model Evaluation")
    print(f"============================================================")
    print(f"Evaluation Set: {args.test_dir} ({len(dataset)} images)")
    print(f"Classes ({num_classes}): {dataset.classes}")
    print(f"Device: {device}")
    print(f"============================================================\n")

    y_true = []
    y_pred = []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    overall_acc = accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"Overall Test Accuracy: {overall_acc:.2f}%")
    print(f"Macro F1-Score:        {macro_f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=dataset.classes, digits=4))

    os.makedirs(args.output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(args.output_dir, "eval_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names=dataset.classes, save_path=cm_path)

    # Save evaluation metrics JSON
    metrics = {
        "accuracy": float(overall_acc),
        "macro_f1": float(macro_f1),
        "total_test_samples": len(dataset),
        "classes": dataset.classes
    }
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved evaluation metrics to: {metrics_path}")
    print(f"Saved confusion matrix plot to: {cm_path}")


if __name__ == "__main__":
    main()
