"""
Industrial Quality Grading Engine for Fruit and Vegetable Quality Assessment.
Combines deep CNN representations (via MobileNetV3 / Custom CNN), Grad-CAM heatmaps,
and Computer Vision surface defect segmentation to handle both studio-synthetic images
and real-world camera photos taken in kitchens, countertops, and markets.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image

CLASS_NAMES = [
    "fresh_apple",
    "rotten_apple",
    "fresh_banana",
    "rotten_banana",
    "fresh_orange",
    "rotten_orange",
    "fresh_tomato",
    "rotten_tomato"
]

COMMODITY_KEYWORDS = {
    "banana": ["banana"],
    "apple": ["apple", "granny smith"],
    "orange": ["orange", "lemon", "citrus"],
    "tomato": ["tomato", "bell pepper", "capsicum"]
}


@dataclass
class QualityReport:
    commodity: str
    condition: str
    grade: str
    grade_description: str
    freshness_score: float
    defect_percentage: float
    confidence: float
    estimated_shelf_life: str
    recommended_action: str
    status_color: str
    probabilities: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


class QualityGrader:
    """
    Industrial Quality Grading Engine.
    Evaluates commodity identity, surface defects, freshness index,
    and assigns industrial grade tiers (Grade A, B, C).
    """
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or CLASS_NAMES
        self._mobilenet = None
        self._weights = None

    def _get_mobilenet(self):
        if self._mobilenet is None:
            self._weights = MobileNet_V3_Small_Weights.DEFAULT
            self._mobilenet = models.mobilenet_v3_small(weights=self._weights).eval()
        return self._mobilenet, self._weights

    def detect_real_commodity(self, pil_image: Image.Image) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Uses pretrained MobileNetV3 backbone to robustly identify the fruit/veg commodity
        on real-world backgrounds (marble tables, kitchen counters, flash glares).
        """
        mobilenet, weights = self._get_mobilenet()
        preprocess = weights.transforms()
        batch = preprocess(pil_image).unsqueeze(0)

        with torch.no_grad():
            logits = mobilenet(batch)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        categories = weights.meta["categories"]
        top_indices = torch.topk(probs, 20).indices.tolist()

        detected_commodity = None
        detected_conf = 0.0

        for idx in top_indices:
            cat_name = categories[idx].lower()
            conf = float(probs[idx].item())
            for target_comm, synonyms in COMMODITY_KEYWORDS.items():
                if any(syn in cat_name for syn in synonyms):
                    if conf > detected_conf:
                        detected_commodity = target_comm.capitalize()
                        detected_conf = conf

        # Format probability distribution
        top5_dict = {}
        for idx in top_indices[:5]:
            cat = categories[idx]
            top5_dict[cat] = round(float(probs[idx].item()), 4)

        return detected_commodity, detected_conf, top5_dict

    def analyze_surface_defects(
        self,
        original_rgb: np.ndarray,
        commodity: str
    ) -> Tuple[float, float, np.ndarray]:
        """
        Extracts fruit body contour and calculates surface defect / blemish coverage.
        Returns:
            defect_percentage (float): 0.0 - 100.0%
            freshness_score (float): 0.0 - 100.0%
            defect_mask (np.ndarray): 2D binary defect mask
        """
        hsv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
        h, w = original_rgb.shape[:2]

        comm = commodity.lower()

        if "banana" in comm:
            # Yellow peel range
            mask_body = cv2.inRange(hsv, np.array([12, 40, 40]), np.array([38, 255, 255]))
            # Dark brown spots / bruising range
            mask_spot = cv2.inRange(hsv, np.array([0, 20, 10]), np.array([30, 255, 120]))
        elif "apple" in comm or "tomato" in comm:
            # Red/Green body
            mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
            mask_red2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
            mask_green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            mask_body = mask_red1 | mask_red2 | mask_green
            mask_spot = cv2.inRange(hsv, np.array([5, 30, 15]), np.array([25, 255, 90]))
        elif "orange" in comm:
            # Orange body
            mask_body = cv2.inRange(hsv, np.array([8, 60, 60]), np.array([24, 255, 255]))
            # Dark mold / rot
            mask_spot = cv2.inRange(hsv, np.array([35, 30, 20]), np.array([120, 255, 150]))
        else:
            gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
            _, mask_body = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask_spot = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))

        # Smooth and find largest contour (the fruit itself, excluding countertop)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_closed = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            fruit_silhouette = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(fruit_silhouette, [largest_contour], -1, 255, thickness=-1)
            fruit_pixels = cv2.countNonZero(fruit_silhouette)

            # Measure defects strictly INSIDE the fruit boundary
            defects_inside = cv2.bitwise_and(mask_spot, fruit_silhouette)
            defect_pixels = cv2.countNonZero(defects_inside)

            defect_pct = float((defect_pixels / max(1, fruit_pixels)) * 100.0)
        else:
            defect_pct = 15.0
            defects_inside = np.zeros((h, w), dtype=np.uint8)

        # Freshness Score computation
        freshness_score = float(max(0.0, min(100.0, 100.0 - (defect_pct * 1.5))))
        return round(defect_pct, 1), round(freshness_score, 1), defects_inside

    def grade(
        self,
        probabilities: np.ndarray,
        heatmap_2d: Optional[np.ndarray] = None,
        original_rgb: Optional[np.ndarray] = None,
        pil_image: Optional[Image.Image] = None
    ) -> QualityReport:
        """
        Computes the full quality grading report.
        Integrates deep feature confidence, Grad-CAM attention,
        and real-world contour defect segmentation.
        """
        # Step 1: Check if real-world commodity can be identified via transfer backbone
        real_comm = None
        real_conf = 0.0
        real_top_probs = {}

        if pil_image is not None:
            try:
                real_comm, real_conf, real_top_probs = self.detect_real_commodity(pil_image)
            except Exception:
                real_comm = None

        pred_idx = int(np.argmax(probabilities))
        pred_class = self.class_names[pred_idx]
        confidence = float(probabilities[pred_idx])

        # If MobileNet detected a known commodity with high confidence (e.g. Banana 93%),
        # prioritize it over a background-shifted custom CNN synthetic prediction
        if real_comm is not None and real_conf > 0.40:
            commodity = real_comm
            confidence = float(real_conf)
        else:
            parts = pred_class.split("_")
            commodity = "_".join(parts[1:]).capitalize()

        # Step 2: Surface defect analysis
        if original_rgb is not None:
            defect_percentage, freshness_score, defect_mask = self.analyze_surface_defects(
                original_rgb, commodity
            )
        else:
            # Fallback to heatmap or prob ratio
            condition_prefix = pred_class.split("_")[0]
            if heatmap_2d is not None and condition_prefix == "rotten":
                defect_mask = heatmap_2d > 0.55
                defect_percentage = float(round(float(np.mean(defect_mask)) * 80.0 + 20.0, 1))
                freshness_score = float(round(max(0.0, 100.0 - defect_percentage), 1))
            else:
                defect_percentage = 4.0 if condition_prefix == "fresh" else 45.0
                freshness_score = 94.0 if condition_prefix == "fresh" else 20.0

        # Step 3: Determine Quality Grade
        # Grade A: Fresh / Premium (Defects < 12%, Freshness >= 82%)
        # Grade B: Secondary / Processing / Ripe (Defects 12% - 32%, Freshness 45% - 81%)
        # Grade C: Substandard / Rotten (Defects > 32%, Freshness < 45%)
        if freshness_score >= 82.0 and defect_percentage < 12.0:
            grade = "Grade A"
            grade_description = "Premium / Export Quality"
            status_color = "#10b981"  # Emerald
            action = "Approved for premium supermarket retail display and export packing."
            shelf_life = self._estimate_shelf_life(commodity, grade="A")
            condition = "Fresh & Firm"
        elif freshness_score >= 45.0 or defect_percentage <= 32.0:
            grade = "Grade B"
            grade_description = "Secondary / Processing Quality (Ripe)"
            status_color = "#f59e0b"  # Amber
            action = "High natural sweetness with minor blemishes. Ideal for smoothies, purées, commercial baking, or immediate discount retail."
            shelf_life = self._estimate_shelf_life(commodity, grade="B")
            condition = "Ripe / Minor Blemishes"
        else:
            grade = "Grade C"
            grade_description = "Substandard / Spoiled"
            status_color = "#ef4444"  # Red
            action = "Unfit for retail sale. Direct to organic composting or bio-fuel conversion."
            shelf_life = "0 Days (Discard immediately)"
            condition = "Rotten / Spoiled"

        # Build clean probabilities dictionary
        prob_dict = {
            f"{commodity.lower()}_fresh": round(float(freshness_score / 100.0), 4),
            f"{commodity.lower()}_rotten": round(float(defect_percentage / 100.0), 4)
        }
        for name, prob in zip(self.class_names, probabilities):
            prob_dict[name] = round(float(prob), 4)

        return QualityReport(
            commodity=commodity,
            condition=condition,
            grade=grade,
            grade_description=grade_description,
            freshness_score=float(freshness_score),
            defect_percentage=float(defect_percentage),
            confidence=float(confidence),
            estimated_shelf_life=shelf_life,
            recommended_action=action,
            status_color=status_color,
            probabilities=prob_dict
        )

    def _estimate_shelf_life(self, commodity: str, grade: str) -> str:
        shelf_data = {
            "apple": {"A": "14 - 21 Days (Cold Storage)", "B": "3 - 5 Days"},
            "banana": {"A": "5 - 7 Days (Ambient)", "B": "1 - 2 Days"},
            "orange": {"A": "10 - 14 Days (Ambient/Cool)", "B": "3 - 4 Days"},
            "tomato": {"A": "7 - 10 Days (Ambient/Cool)", "B": "2 - 3 Days"},
        }
        fallback = {"A": "7 - 12 Days", "B": "2 - 4 Days"}
        return shelf_data.get(commodity.lower(), fallback).get(grade, "2 - 3 Days")
