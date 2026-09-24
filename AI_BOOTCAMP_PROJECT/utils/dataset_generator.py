"""
Synthetic and Benchmark Dataset Generator for Fruit & Vegetable Quality Grading.
Generates structured image datasets with realistic textures, colors, shapes,
and defect patterns (bruising, mold, rot) so that the CNN pipeline can be trained,
evaluated, and demonstrated immediately.
"""

import os
import random
from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASSES = [
    "fresh_apple",
    "rotten_apple",
    "fresh_banana",
    "rotten_banana",
    "fresh_orange",
    "rotten_orange",
    "fresh_tomato",
    "rotten_tomato"
]


def _create_fruit_canvas(size: Tuple[int, int] = (128, 128)) -> Tuple[Image.Image, ImageDraw.Draw]:
    """Generates an image with varied realistic backgrounds: white counter, dark marble, wood desk."""
    bg_style = random.random()
    if bg_style < 0.40:
        # Dark marble / slate countertop (like kitchen tables)
        base = random.randint(18, 50)
        bg = np.full((size[1], size[0], 3), base, dtype=np.uint8)
        # White / gray marble veining
        veins = np.random.choice([0, 1], size=bg.shape[:2], p=[0.97, 0.03])
        bg[veins == 1] = np.random.randint(90, 160, size=(np.sum(veins == 1), 3), dtype=np.uint8)
    elif bg_style < 0.70:
        # Bright / light supermarket shelf
        base = random.randint(210, 250)
        bg = np.full((size[1], size[0], 3), base, dtype=np.uint8)
    else:
        # Warm wooden surface / cutting board
        bg = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        bg[:, :, 0] = random.randint(130, 170)  # R
        bg[:, :, 1] = random.randint(85, 115)   # G
        bg[:, :, 2] = random.randint(50, 75)    # B

    noise = np.random.randint(-8, 8, bg.shape, dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)
    return img, draw


def generate_apple_image(fresh: bool, size: Tuple[int, int] = (128, 128)) -> Image.Image:
    img, draw = _create_fruit_canvas(size)
    cx, cy = size[0] // 2 + random.randint(-4, 4), size[1] // 2 + random.randint(-3, 5)
    r = random.randint(38, 46)

    # Base apple color (red with subtle yellow/green hue)
    if random.random() > 0.3:
        base_color = (random.randint(185, 220), random.randint(25, 50), random.randint(25, 50))
    else:
        # Green granny smith apple
        base_color = (random.randint(110, 150), random.randint(180, 220), random.randint(30, 60))

    # Apple silhouette (slightly indented at top and bottom)
    bbox = [cx - r, cy - r, cx + r, cy + r]
    draw.ellipse(bbox, fill=base_color, outline=(base_color[0] - 30, max(0, base_color[1] - 20), max(0, base_color[2] - 20)))

    # Stem
    draw.line([(cx, cy - r + 3), (cx + random.randint(-4, 4), cy - r - 12)], fill=(80, 50, 25), width=3)

    # Specular highlight for fresh apple
    if fresh:
        hl_box = [cx - r // 2, cy - r // 2, cx - r // 4, cy - r // 4]
        draw.ellipse(hl_box, fill=(min(255, base_color[0] + 50), min(255, base_color[1] + 50), min(255, base_color[2] + 50)))
    else:
        # Rotten: multiple dark brown decay spots and fungal discoloration
        num_spots = random.randint(3, 7)
        for _ in range(num_spots):
            sx = cx + random.randint(-r + 10, r - 10)
            sy = cy + random.randint(-r + 10, r - 10)
            sr = random.randint(8, 18)
            decay_color = (random.randint(55, 85), random.randint(35, 55), random.randint(20, 35))
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=decay_color)
            # Mold center
            if random.random() > 0.4:
                draw.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], fill=(180, 195, 175))

    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img


def generate_banana_image(fresh: bool, size: Tuple[int, int] = (128, 128)) -> Image.Image:
    img, draw = _create_fruit_canvas(size)
    cx, cy = size[0] // 2, size[1] // 2

    # Banana curved polygon
    if fresh:
        body_color = (random.randint(235, 255), random.randint(210, 235), random.randint(40, 70))
        tip_color = (random.randint(120, 150), random.randint(165, 195), random.randint(30, 50))
    else:
        # Rotten banana: darkened brown/black body
        body_color = (random.randint(100, 140), random.randint(75, 105), random.randint(35, 55))
        tip_color = (40, 30, 20)

    # Draw curved banana body using wide polygon / arc
    points = [
        (cx - 45, cy + 20),
        (cx - 20, cy - 25),
        (cx + 25, cy - 25),
        (cx + 45, cy + 15),
        (cx + 40, cy + 25),
        (cx + 15, cy - 8),
        (cx - 25, cy - 8),
        (cx - 42, cy + 28),
    ]
    draw.polygon(points, fill=body_color)
    draw.line([(cx - 45, cy + 20), (cx - 52, cy + 30)], fill=tip_color, width=4)
    draw.line([(cx + 45, cy + 15), (cx + 52, cy + 22)], fill=tip_color, width=4)

    if not fresh:
        # Add brown-black necrotic spots
        for _ in range(random.randint(5, 12)):
            bx = random.randint(cx - 35, cx + 35)
            by = random.randint(cy - 20, cy + 15)
            br = random.randint(3, 7)
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(45, 30, 20))

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Real-world orientation diversity: rotate at varied angles
    angle = random.choice([0, 30, 60, 90, 120, 180, 240, 270, 315])
    if angle != 0:
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR)

    return img


def generate_orange_image(fresh: bool, size: Tuple[int, int] = (128, 128)) -> Image.Image:
    img, draw = _create_fruit_canvas(size)
    cx, cy = size[0] // 2 + random.randint(-3, 3), size[1] // 2 + random.randint(-3, 3)
    r = random.randint(38, 44)

    if fresh:
        base_color = (random.randint(235, 255), random.randint(125, 155), random.randint(10, 30))
    else:
        # Dull brownish-orange
        base_color = (random.randint(150, 180), random.randint(85, 110), random.randint(25, 45))

    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=base_color)

    # Orange peel texture / stippling
    for _ in range(60):
        px = cx + random.randint(-r + 6, r - 6)
        py = cy + random.randint(-r + 6, r - 6)
        if (px - cx) ** 2 + (py - cy) ** 2 < (r - 4) ** 2:
            draw.point((px, py), fill=(max(0, base_color[0] - 25), max(0, base_color[1] - 25), base_color[2]))

    if not fresh:
        # Blue-green mold patch (Penicillium digitatum)
        mx = cx + random.randint(-15, 15)
        my = cy + random.randint(-15, 15)
        mr = random.randint(12, 22)
        # Mold ring
        draw.ellipse([mx - mr, my - mr, mx + mr, my + mr], fill=(60, 110, 95))
        draw.ellipse([mx - (mr - 4), my - (mr - 4), mx + (mr - 4), my + (mr - 4)], fill=(190, 220, 210))
        draw.ellipse([mx - (mr - 8), my - (mr - 8), mx + (mr - 8), my + (mr - 8)], fill=(45, 80, 70))

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def generate_tomato_image(fresh: bool, size: Tuple[int, int] = (128, 128)) -> Image.Image:
    img, draw = _create_fruit_canvas(size)
    cx, cy = size[0] // 2 + random.randint(-3, 3), size[1] // 2 + random.randint(-2, 4)
    r = random.randint(37, 45)

    if fresh:
        base_color = (random.randint(215, 245), random.randint(30, 50), random.randint(30, 45))
    else:
        # Discolored soft watery rot
        base_color = (random.randint(130, 160), random.randint(45, 65), random.randint(35, 50))

    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=base_color)

    # Green calyx / star on top
    calyx_color = (35, 130, 45) if fresh else (65, 85, 45)
    for angle in [0, 60, 120, 180, 240, 300]:
        rad = np.deg2rad(angle)
        ex = cx + int(14 * np.cos(rad))
        ey = (cy - r + 4) + int(14 * np.sin(rad))
        draw.line([(cx, cy - r + 4), (ex, ey)], fill=calyx_color, width=3)

    if fresh:
        # Glossy highlight
        draw.ellipse([cx - r // 2, cy - r // 2, cx - r // 4, cy - r // 4], fill=(255, 120, 120))
    else:
        # Wrinkled lesion / sunken brown necrosis
        for _ in range(random.randint(3, 6)):
            sx = cx + random.randint(-r + 10, r - 10)
            sy = cy + random.randint(-r + 15, r - 10)
            sr = random.randint(8, 15)
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(50, 35, 25))

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def generate_sample_for_class(class_name: str, size: Tuple[int, int] = (128, 128)) -> Image.Image:
    """Generates a synthetic sample image for the specified class name."""
    is_fresh = class_name.startswith("fresh_")
    if "apple" in class_name:
        return generate_apple_image(is_fresh, size)
    elif "banana" in class_name:
        return generate_banana_image(is_fresh, size)
    elif "orange" in class_name:
        return generate_orange_image(is_fresh, size)
    elif "tomato" in class_name:
        return generate_tomato_image(is_fresh, size)
    else:
        raise ValueError(f"Unknown commodity class: {class_name}")


def create_dataset(
    output_dir: str = "data/dataset",
    train_per_class: int = 60,
    val_per_class: int = 20,
    size: Tuple[int, int] = (128, 128)
):
    """
    Creates complete train and validation datasets ready for ImageFolder.
    """
    for split, count in [("train", train_per_class), ("val", val_per_class)]:
        for c in CLASSES:
            folder = os.path.join(output_dir, split, c)
            os.makedirs(folder, exist_ok=True)
            for i in range(count):
                img = generate_sample_for_class(c, size=size)
                filepath = os.path.join(folder, f"{c}_{i:04d}.png")
                img.save(filepath, "PNG")

    print(f"Generated dataset at: {output_dir}")
    print(f"Train samples per class: {train_per_class} (Total: {train_per_class * len(CLASSES)})")
    print(f"Val samples per class: {val_per_class} (Total: {val_per_class * len(CLASSES)})")

    # Also create stand-alone sample test images
    sample_dir = os.path.join("data", "samples")
    os.makedirs(sample_dir, exist_ok=True)
    for c in CLASSES:
        sample_img = generate_sample_for_class(c, size=size)
        sample_path = os.path.join(sample_dir, f"{c}_test.png")
        sample_img.save(sample_path, "PNG")
    print(f"Created reference test samples in: {sample_dir}")


if __name__ == "__main__":
    create_dataset(output_dir="data/dataset", train_per_class=60, val_per_class=20)
