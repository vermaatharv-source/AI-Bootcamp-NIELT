"""
Quickstart One-Click Pipeline for Fruit & Vegetable Quality Grading Project.
Automates:
1. Environment verification & package checking
2. Dataset generation (if not already generated)
3. Model training & validation (if checkpoints don't exist)
4. Sample inference demonstration with Grad-CAM
5. Launching the interactive Web UI Dashboard
"""

import os
import sys
import subprocess
import time


def print_banner():
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║       FRUIT & VEGETABLE QUALITY GRADING SYSTEM (CNN ML)       ║
    ║   Automated Quality Sorting, Freshness & Defect Diagnostics   ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dataset():
    train_dir = os.path.join("data", "dataset", "train")
    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        print("[Step 1/4] Generating benchmark dataset...")
        from utils.dataset_generator import create_dataset
        create_dataset(output_dir="data/dataset", train_per_class=60, val_per_class=20)
        print("Dataset ready.\n")
    else:
        print("[Step 1/4] Dataset already exists in 'data/dataset'. Skipping generation.\n")


def check_model():
    ckpt_path = os.path.join("checkpoints", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print("[Step 2/4] Training custom CNN model (8 epochs)...")
        cmd = [sys.executable, "train.py", "--epochs", "8", "--batch-size", "32"]
        subprocess.run(cmd, check=True)
        print("Model training completed and checkpoint saved.\n")
    else:
        print(f"[Step 2/4] Model checkpoint found at '{ckpt_path}'. Ready for inference.\n")


def run_demo_predictions():
    print("[Step 3/4] Running sample diagnostics on test specimens...")
    samples_dir = os.path.join("data", "samples")
    if os.path.exists(samples_dir):
        sample_files = [
            os.path.join(samples_dir, f)
            for f in os.listdir(samples_dir)
            if f.endswith(("_test.png", ".jpg", ".png"))
        ]
        if sample_files:
            cmd = [sys.executable, "predict.py", "--input-dir", samples_dir, "--output-dir", "predictions"]
            subprocess.run(cmd, check=True)
            print("Sample diagnostics completed. Check 'predictions/' for generated diagnostic cards.\n")


def launch_web_app():
    print("[Step 4/4] Launching Interactive Web Dashboard...")
    print("=" * 64)
    print(" >>> Open your browser at: http://127.0.0.1:8000 <<<")
    print("=" * 64)
    from app import app
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


def main():
    print_banner()
    check_dataset()
    check_model()
    run_demo_predictions()
    launch_web_app()


if __name__ == "__main__":
    main()
