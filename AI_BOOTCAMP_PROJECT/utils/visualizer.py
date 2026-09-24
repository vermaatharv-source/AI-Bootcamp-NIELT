"""
Visualization utilities for model metrics, confusion matrix, and visual reports.
"""

from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / server environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = "checkpoints/training_curves.png"
):
    """
    Plots training and validation Loss and Accuracy across epochs.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 5))

    # Loss Subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#3b82f6", linewidth=2)
    plt.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ef4444", linewidth=2)
    plt.title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)

    # Accuracy Subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "o-", label="Train Accuracy", color="#10b981", linewidth=2)
    plt.plot(epochs, history["val_acc"], "s--", label="Val Accuracy", color="#8b5cf6", linewidth=2)
    plt.title("Classification Accuracy vs. Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str = "checkpoints/confusion_matrix.png"
):
    """
    Visualizes normalized Confusion Matrix using Seaborn heatmap.
    """
    # Normalize by row (actual class totals)
    with np.errstate(all='ignore'):
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    clean_names = [c.replace("_", " ").title() for c in class_names]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=clean_names,
        yticklabels=clean_names,
        cbar_kws={"label": "Normalized Ratio"}
    )
    plt.title("Normalized Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Ground Truth Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def generate_prediction_card(
    original_rgb: np.ndarray,
    heatmap_overlay_rgb: np.ndarray,
    report,
    save_path: str
):
    """
    Creates a visual side-by-side diagnostic card:
    [Original Image] | [Grad-CAM Defect Attention] | [Quality Grade Badge & Stats]
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    axes[0].imshow(original_rgb)
    axes[0].set_title(f"Input: {report.commodity}", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(heatmap_overlay_rgb)
    axes[1].set_title("Grad-CAM XAI Attention", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.suptitle(
        f"{report.grade}: {report.condition} (Freshness: {report.freshness_score}%)",
        fontsize=14,
        fontweight="bold",
        color=report.status_color
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
