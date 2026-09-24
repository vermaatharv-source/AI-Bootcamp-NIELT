"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN Explainability.
Visualizes which regions of a fruit or vegetable image led the CNN model
to classify it as fresh or defective/rotten.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """
    Computes Grad-CAM heatmaps for any PyTorch CNN model.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            if hasattr(model, "get_target_layer_for_gradcam"):
                self.target_layer = model.get_target_layer_for_gradcam()
            else:
                # Fallback: grab last Conv2d layer
                conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                if not conv_layers:
                    raise ValueError("No Conv2d layers found in model.")
                self.target_layer = conv_layers[-1]
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generates a 2D heatmap [0.0, 1.0] corresponding to input_tensor.
        input_tensor: torch.Tensor with shape (1, 3, H, W).
        target_class: index of class. If None, highest probability class is used.
        """
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        # Backward pass for the target class score
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Gradients: (1, C, H_feat, W_feat), Activations: (1, C, H_feat, W_feat)
        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients across spatial dimensions
        alpha = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(alpha * activations, dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)
        cam = torch.relu(cam)  # Discard negative influences

        cam_np = cam.squeeze().cpu().numpy()

        # Normalize heatmap between 0 and 1
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, target_class

    def overlay_heatmap(
        self,
        original_image_rgb: np.ndarray,
        heatmap_2d: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlays a normalized 2D heatmap onto the original RGB image.
        original_image_rgb: (H, W, 3) uint8 RGB image
        heatmap_2d: (H_feat, W_feat) float32 in [0, 1]
        Returns: (H, W, 3) blended RGB image uint8
        """
        h, w = original_image_rgb.shape[:2]
        resized_heatmap = cv2.resize(heatmap_2d, (w, h))

        # Convert to 0-255 uint8
        heatmap_uint8 = np.uint8(255 * resized_heatmap)

        # Apply OpenCV colormap (returns BGR)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Alpha blend
        blended = np.clip(
            alpha * heatmap_rgb.astype(np.float32) + (1.0 - alpha) * original_image_rgb.astype(np.float32),
            0,
            255
        ).astype(np.uint8)

        return blended
