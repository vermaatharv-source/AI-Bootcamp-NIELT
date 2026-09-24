"""
Data Augmentation, Normalization, and Preprocessing Pipelines.
"""

from typing import Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: Tuple[int, int] = (128, 128)) -> T.Compose:
    """
    Data augmentation pipeline for training the CNN model.
    Applies spatial, geometric, and color perturbations to prevent overfitting.
    """
    return T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=25),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms(image_size: Tuple[int, int] = (128, 128)) -> T.Compose:
    """
    Validation and test evaluation pipeline (deterministic, no augmentation).
    """
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def preprocess_image_for_model(
    image_input: Union[str, Image.Image, np.ndarray],
    image_size: Tuple[int, int] = (128, 128),
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Takes an image (filepath, PIL Image, or numpy array), converts to RGB,
    resizes, normalizes for the model, and returns:
      1. input_tensor: torch.Tensor of shape (1, 3, H, W) on the target device.
      2. original_rgb: np.ndarray uint8 of shape (H, W, 3) for visualization.
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        # Assume RGB or BGR? If uint8 3-channel
        if image_input.shape[2] == 4:
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            pil_img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Resize original for display
    display_pil = pil_img.resize(image_size, Image.Resampling.BILINEAR)
    original_rgb = np.array(display_pil, dtype=np.uint8)

    val_tf = get_val_transforms(image_size)
    tensor = val_tf(display_pil).unsqueeze(0).to(device)

    return tensor, original_rgb
