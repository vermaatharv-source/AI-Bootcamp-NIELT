"""
FastAPI Web Application and REST API for Fruit & Vegetable Quality Grading CNN.
Provides an interactive web dashboard with:
- Drag & drop image upload
- Preset sample fruit/vegetable selector
- Webcam capture support
- Real-time CNN inference
- Grad-CAM Explainable AI defect visualization
- Industrial quality grading metrics (Grade A/B/C, Freshness Index, Shelf-life)
"""

import base64
import io
import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from models.cnn_model import build_model
from models.gradcam import GradCAM
from models.quality_grader import QualityGrader
from utils.transforms import preprocess_image_for_model

app = FastAPI(
    title="Fruit & Vegetable Quality Grading System",
    description="AI-powered CNN system for optical fruit & vegetable freshness sorting and quality grading with Grad-CAM Explainability.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded model
MODEL = None
CLASS_NAMES: List[str] = []
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRADER = None
IMAGE_SIZE = 128


def load_model_if_available():
    global MODEL, CLASS_NAMES, GRADER, IMAGE_SIZE
    ckpt_path = "checkpoints/best_model.pth"
    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint '{ckpt_path}' not found yet. Web app will run; please train model or generate quickstart.")
        return

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    CLASS_NAMES = checkpoint.get("class_names", [
        "fresh_apple", "rotten_apple",
        "fresh_banana", "rotten_banana",
        "fresh_orange", "rotten_orange",
        "fresh_tomato", "rotten_tomato"
    ])
    IMAGE_SIZE = checkpoint.get("image_size", 128)
    model_type = checkpoint.get("model_type", "custom_cnn")

    MODEL = build_model(num_classes=len(CLASS_NAMES), model_type=model_type)
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL.to(DEVICE)
    MODEL.eval()

    GRADER = QualityGrader(class_names=CLASS_NAMES)
    print(f"Successfully loaded {model_type} from {ckpt_path} on {DEVICE}!")


# Load model at startup
load_model_if_available()


class PredictRequest(BaseModel):
    image: str  # Base64 data URI or raw base64 string


def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


@app.get("/api/model-info")
def get_model_info():
    """Returns metadata about the active model and trained classes."""
    summary_path = "checkpoints/training_summary.json"
    summary_data = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary_data = json.load(f)

    return {
        "status": "loaded" if MODEL is not None else "not_loaded",
        "device": str(DEVICE),
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "input_resolution": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "metrics": summary_data
    }


@app.get("/api/samples")
def list_sample_images():
    """Lists available pre-generated fruit/vegetable test samples."""
    samples_dir = "data/samples"
    if not os.path.exists(samples_dir):
        return {"samples": []}

    samples = [
        f for f in sorted(os.listdir(samples_dir))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return {"samples": samples}


@app.get("/api/sample/{sample_name}")
def get_sample_image(sample_name: str):
    """Returns the base64 encoded image for a sample item."""
    path = os.path.join("data/samples", os.path.basename(sample_name))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample image not found.")

    with Image.open(path) as img:
        b64 = pil_to_base64(img)
    return {"image": b64, "name": sample_name}


@app.post("/api/predict")
def predict_quality(request: PredictRequest):
    """
    Accepts a base64 encoded image, runs CNN prediction and Grad-CAM,
    and returns full industrial quality grading metrics and heatmaps.
    """
    global MODEL, GRADER, CLASS_NAMES

    if MODEL is None:
        load_model_if_available()
        if MODEL is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not yet trained or loaded. Please run 'python train.py' or 'python quickstart.py' first."
            )

    # Decode base64 image
    raw_b64 = request.image
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_b64)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    # Preprocess
    tensor, original_rgb = preprocess_image_for_model(
        pil_image,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        device=DEVICE
    )

    # CNN Inference
    with torch.no_grad():
        logits = MODEL(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    # Grad-CAM Attention Map
    gradcam = GradCAM(MODEL)
    heatmap_2d, _ = gradcam.generate_heatmap(tensor)
    gradcam.remove_hooks()

    # Blend Heatmap with original
    overlay_rgb = gradcam.overlay_heatmap(original_rgb, heatmap_2d, alpha=0.55)

    # Grade assessment
    report = GRADER.grade(
        probabilities=probs,
        heatmap_2d=heatmap_2d,
        original_rgb=original_rgb,
        pil_image=pil_image
    )

    # Convert images to base64 for client display
    orig_b64 = pil_to_base64(Image.fromarray(original_rgb))
    heatmap_b64 = pil_to_base64(Image.fromarray(overlay_rgb))

    return {
        "report": report.to_dict(),
        "original_image": orig_b64,
        "gradcam_heatmap": heatmap_b64,
        "input_resolution": f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    }


# Serve static web frontend
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    """Serves the main interactive dashboard."""
    html_path = "templates/index.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Fruit & Vegetable Quality Grading System API is running.</h2>")


if __name__ == "__main__":
    print("Starting Fruit & Vegetable Quality Grading Web Server at http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
