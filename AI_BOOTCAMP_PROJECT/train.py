"""
Training Pipeline for Fruit & Vegetable Quality Grading CNN.
Features:
- Custom FruitQualityCNN or Transfer Learning (MobileNetV3 / ResNet18)
- Data augmentation & regularization
- Learning rate scheduling & early stopping
- Automatic checkpointing and evaluation metrics (Loss, Accuracy, F1, Confusion Matrix)
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from models.cnn_model import build_model
from utils.transforms import get_train_transforms, get_val_transforms
from utils.visualizer import plot_training_history, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN Quality Grading Model")
    parser.add_argument("--data-dir", type=str, default="data/dataset", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization weight decay")
    parser.add_argument("--model-type", type=str, default="custom_cnn", choices=["custom_cnn", "mobilenet_v3", "resnet18"], help="Model architecture")
    parser.add_argument("--image-size", type=int, default=128, help="Input image dimension (H=W)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save model weights")
    parser.add_argument("--patience", type=int, default=6, help="Patience for early stopping")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    val_loss = running_loss / total
    val_acc = (correct / total) * 100.0
    return val_loss, val_acc, np.array(all_targets), np.array(all_preds)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f" Fruit & Vegetable Quality Grading CNN Training Pipeline")
    print(f"============================================================")
    print(f"Device:           {device}")
    print(f"Model:            {args.model_type}")
    print(f"Image Size:       {args.image_size}x{args.image_size}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Data Directory:   {args.data_dir}")
    print(f"Checkpoints Dir:  {args.checkpoint_dir}")
    print(f"============================================================")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset Paths
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Error: Dataset directory '{args.data_dir}' must contain 'train' and 'val' subfolders.")
        print("Run 'python utils/dataset_generator.py' to generate default dataset.")
        sys.exit(1)

    image_size = (args.image_size, args.image_size)
    train_transforms = get_train_transforms(image_size)
    val_transforms = get_val_transforms(image_size)

    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    val_dataset = ImageFolder(val_dir, transform=val_transforms)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Save class names mapping
    class_map_path = os.path.join(args.checkpoint_dir, "class_names.json")
    with open(class_map_path, "w") as f:
        json.dump(class_names, f, indent=2)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Instantiate Model
    model = build_model(num_classes=num_classes, model_type=args.model_type)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Tracking metrics
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_weights_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    start_time = time.time()

    print("\n--- Commencing Model Training ---")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - t0
        lr_curr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}% | "
            f"LR: {lr_curr:.5f} | Time: {epoch_time:.1f}s"
        )

        # Checkpoint Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_type": args.model_type,
                "num_classes": num_classes,
                "class_names": class_names,
                "best_val_acc": best_val_acc,
                "image_size": args.image_size
            }, best_weights_path)
            print(f"  >>> Checkpoint: Best validation accuracy updated ({best_val_acc:.2f}%). Saved to {best_weights_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping Triggered] No improvement for {args.patience} epochs.")
                break

    total_duration = time.time() - start_time
    print(f"\nTraining completed in {total_duration / 60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")

    # Load best weights for final evaluation
    checkpoint = torch.load(best_weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, val_acc, y_true, y_pred = validate(model, val_loader, criterion, device)

    # Classification Metrics & Report
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print(f"\n--- Final Best Model Evaluation ---")
    print(f"Validation F1-Score (Macro): {f1_macro:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Generate and save plots
    curves_path = os.path.join(args.checkpoint_dir, "training_curves.png")
    plot_training_history(history, save_path=curves_path)

    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(args.checkpoint_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names=class_names, save_path=cm_path)

    # Save training summary JSON
    summary = {
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_acc),
        "f1_score_macro": float(f1_macro),
        "total_epochs_trained": len(history["train_loss"]),
        "model_type": args.model_type,
        "num_classes": num_classes,
        "classes": class_names
    }
    with open(os.path.join(args.checkpoint_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining and evaluation pipeline finished successfully!")


if __name__ == "__main__":
    main()
