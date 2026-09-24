"""
Deep Convolutional Neural Network (CNN) for Fruit and Vegetable Quality Grading.
Supports custom deep CNN architecture and transfer learning backbones.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    """
    Dual Convolutional Block with Batch Normalization, ReLU activation,
    Max Pooling, and Dropout for regularization.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.2):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FruitQualityCNN(nn.Module):
    """
    Custom Deep CNN Architecture for Fruit & Vegetable Quality Grading.
    Comprises 4 hierarchical Convolutional stages extracting low to high-level features:
    - Block 1 (3 -> 32 channels): Edge and color gradient detection
    - Block 2 (32 -> 64 channels): Texture and surface contour representation
    - Block 3 (64 -> 128 channels): Defect patterns, bruising, fungal mold spots
    - Block 4 (128 -> 256 channels): High-level semantic quality representation
    Followed by Adaptive Average Pooling and a Fully Connected Classifier with Dropout.
    """
    def __init__(self, num_classes: int = 8, in_channels: int = 3):
        super(FruitQualityCNN, self).__init__()
        self.num_classes = num_classes

        # Feature Extractor Blocks
        self.block1 = ConvBlock(in_channels, 32, dropout_rate=0.15)
        self.block2 = ConvBlock(32, 64, dropout_rate=0.20)
        self.block3 = ConvBlock(64, 128, dropout_rate=0.25)
        
        # Last Conv Stage (Target layer for Grad-CAM Explainability)
        self.conv_final = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.pool_final = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

    def get_target_layer_for_gradcam(self) -> nn.Module:
        """Returns the final convolutional layer for Grad-CAM gradient hooks."""
        return self.conv_final[3]  # The second Conv2d layer in conv_final

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass up to the final convolutional feature map."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.conv_final(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        x = self.pool_final(feat)
        x = self.global_pool(x)
        out = self.classifier(x)
        return out


def build_model(
    num_classes: int = 8,
    model_type: str = "custom_cnn",
    pretrained: bool = False
) -> nn.Module:
    """
    Factory function to instantiate either the custom FruitQualityCNN
    or transfer-learning backbones (MobileNetV3 / ResNet18).
    """
    model_type = model_type.lower()

    if model_type == "custom_cnn":
        return FruitQualityCNN(num_classes=num_classes)

    elif model_type == "mobilenet_v3":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    elif model_type == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Choose from 'custom_cnn', 'mobilenet_v3', 'resnet18'."
        )


if __name__ == "__main__":
    # Test model instantiation and forward pass
    model = build_model(num_classes=8, model_type="custom_cnn")
    dummy_input = torch.randn(2, 3, 128, 128)
    output = model(dummy_input)
    print("FruitQualityCNN initialized successfully!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape (logits): {output.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
