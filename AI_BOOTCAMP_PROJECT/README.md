# 🍎 AgriGrade AI: Deep CNN Fruit & Vegetable Quality Grading System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CNN%20%2B%20Grad--CAM-blueviolet.svg)]()
[![Status](https://img.shields.io/badge/Project-Ready%20to%20Deploy-success.svg)]()

An end-to-end Machine Learning and Computer Vision system for **automated fruit and vegetable quality grading, freshness detection, and defect diagnosis** using a custom Deep Convolutional Neural Network (CNN) and Explainable AI (Grad-CAM).

---

## 🌟 Key Highlights

- **Custom Deep CNN Architecture**: 4-stage hierarchical convolutional network with Batch Normalization, ReLU, Max Pooling, and Dropout for regularization (~1.2M parameters).
- **Industrial Quality Grading Engine**: Automatically sorts produce into actionable grades:
  - 🟢 **Grade A (Premium / Export Quality)**: High freshness index (≥ 80%), zero or negligible surface blemishes.
  - 🟡 **Grade B (Secondary / Processing Quality)**: Moderate freshness (45% - 79%), minor surface defects; suitable for juices, purées, and bakeries.
  - 🔴 **Grade C (Substandard / Spoiled)**: Active decay, mold, or rot; direct to compost or bio-fuel conversion.
- **Explainable AI (Grad-CAM)**: Visualizes exact convolutional feature activation maps with heatmap overlays, isolating necrotic tissue, fungal spots, and surface bruising.
- **Interactive Web Dashboard**: Modern UI with drag-and-drop upload, webcam snapshot mode, preset test specimens, live gauge meters, and probability breakdowns.
- **Built-in Dataset Generator**: Includes a procedural synthetic dataset generator so you can test, train, and demonstrate immediately without large downloads.
- **Ready for Real-World Datasets**: Fully compatible with standard datasets like Kaggle's *Fresh and Stale Fruits & Vegetables*.

---

## 📁 Project Structure

```text
AI_BOOTCAMP_PROJECT/
├── data/
│   ├── dataset/
│   │   ├── train/                 # Training dataset (per-class subfolders)
│   │   └── val/                   # Validation dataset
│   └── samples/                   # Reference test images for quick testing
├── models/
│   ├── __init__.py
│   ├── cnn_model.py               # Custom FruitQualityCNN + Transfer Learning
│   ├── gradcam.py                 # Grad-CAM Explainable AI implementation
│   └── quality_grader.py          # Industrial Quality Grading & Shelf-Life Engine
├── checkpoints/
│   ├── best_model.pth             # Saved PyTorch model weights
│   ├── class_names.json           # Class label index mapping
│   ├── training_curves.png        # Training & validation Loss/Accuracy plots
│   └── confusion_matrix.png       # Normalized Confusion Matrix heatmap
├── static/
│   ├── css/style.css              # Modern responsive dark-mode styling
│   └── js/main.js                 # Frontend interactive logic & API calls
├── templates/
│   └── index.html                 # Web dashboard interface
├── utils/
│   ├── __init__.py
│   ├── dataset_generator.py       # Procedural realistic fruit/veg image generator
│   ├── transforms.py              # Data augmentations & normalization
│   └── visualizer.py              # Matplotlib & Seaborn diagnostic plotting
├── app.py                         # FastAPI Web Application & REST API
├── train.py                       # CNN Training pipeline with scheduler & early stopping
├── evaluate.py                    # Model evaluation (Accuracy, F1, Confusion Matrix)
├── predict.py                     # Single/Batch CLI inference with Grad-CAM output
├── quickstart.py                  # 1-Click setup: data, training, demo, & web app
├── exploration_and_training.ipynb # Jupyter Notebook for bootcamp submission
└── requirements.txt               # Project dependencies
```

---

## 🚀 Quickstart (1-Click Run)

To automatically verify data, check model weights, run sample predictions, and launch the web dashboard:

```bash
python quickstart.py
```

Then open your browser at **`http://127.0.0.1:8000`** to test produce quality with the interactive web interface!

---

## 🛠️ Step-by-Step Execution Guide

### 1. (Optional) Generate or Refresh Dataset
If you want to create or recreate the built-in benchmark dataset:
```bash
python utils/dataset_generator.py
```
*Creates 480 training images, 160 validation images, and reference test samples in `data/samples/`.*

### 2. Train the CNN Model
Train the custom Deep CNN with data augmentations, Cosine Annealing learning rate scheduling, and early stopping:
```bash
python train.py --epochs 12 --batch-size 32 --lr 0.001
```
Key Arguments:
- `--epochs`: Number of training iterations (default: 15)
- `--batch-size`: Batch size (default: 16)
- `--model-type`: `custom_cnn`, `mobilenet_v3`, or `resnet18` (default: `custom_cnn`)
- `--image-size`: Input resolution (default: 128)

*Outputs saved to `checkpoints/best_model.pth`, `training_curves.png`, and `confusion_matrix.png`.*

### 3. Evaluate the Model
Compute overall test accuracy, macro F1-score, per-class precision/recall, and a confusion matrix:
```bash
python evaluate.py --test-dir data/dataset/val --checkpoint checkpoints/best_model.pth
```

### 4. Run CLI Predictions with Grad-CAM
Inspect a single image:
```bash
python predict.py --image data/samples/fresh_apple_test.png
```
Or run batch inference over an entire directory:
```bash
python predict.py --input-dir data/samples --output-dir predictions
```
*Generated side-by-side diagnostic cards (Original vs Grad-CAM Defect Attention) are saved in `predictions/`.*

### 5. Launch the Web Dashboard
```bash
python app.py
```
Open **`http://127.0.0.1:8000`** in your browser. You can:
- Drag and drop your own fruit/vegetable photos
- Click any of the pre-loaded sample specimens
- Use your webcam to take a live photo
- Inspect the **Grad-CAM attention heatmap** and toggle views
- View the complete industrial quality report, defect percentage, and shelf-life estimate

---

## 📊 Industrial Quality Grading Criteria

| Grade | Classification | Freshness Score | Defect Coverage | Actionable Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Grade A** | Premium / Export Quality | $\ge 80\%$ | $< 5\%$ | Approved for premium supermarket retail and export packing. |
| **Grade B** | Secondary / Processing | $45\% - 79\%$ | $5\% - 25\%$ | Redirect to processing for juices, purées, canning, or bakery. |
| **Grade C** | Substandard / Spoiled | $< 45\%$ | $> 25\%$ | Reject for consumption. Divert to compost or bio-energy. |

---

## 🔬 Explainable AI: Grad-CAM Workflow

1. Forward propagates the image through the convolutional blocks.
2. Extracts feature maps $A^k$ from the final convolutional stage (`conv_final`).
3. Computes the gradient of the winning class score $y^c$ with respect to each activation map: $\frac{\partial y^c}{\partial A^k}$.
4. Applies Global Average Pooling to the gradients to obtain importance weights $\alpha_k^c$.
5. Computes the weighted combination of feature activations followed by a ReLU non-linearity:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
6. Resizes the heatmap to image dimensions and blends it using OpenCV Jet colormap.

---

## 📦 Using Real Datasets (e.g. Kaggle)

To train on real-world datasets like Kaggle's *Fresh and Stale Fruits & Vegetables*:
1. Structure the dataset directory as follows:
   ```text
   data/dataset/
   ├── train/
   │   ├── fresh_apple/
   │   ├── rotten_apple/
   │   └── ...
   └── val/
       ├── fresh_apple/
       ├── rotten_apple/
       └── ...
   ```
2. Run `python train.py --epochs 20 --batch-size 32`.

---

## 🎓 Bootcamp / Academic Presentation Guide

- Open [`exploration_and_training.ipynb`](exploration_and_training.ipynb) in Jupyter / VS Code / Antigravity to step through the code cell by cell.
- Demonstrate the live web dashboard with preset samples and webcam to showcase end-to-end deployment readiness.
- Highlight the **Grad-CAM Explainable AI** component as proof that the CNN learns genuine defect features rather than background noise.
