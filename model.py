import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

BACKBONES = ("efficientnet_b0", "resnet18")


def build_model(
    num_classes: int,
    backbone: str = "efficientnet_b0",
    freeze: bool = True,
) -> nn.Module:
    """Return a pretrained model with its classifier head replaced for num_classes."""
    if backbone == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif backbone == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"backbone must be one of {BACKBONES}, got '{backbone}'")

    return model
