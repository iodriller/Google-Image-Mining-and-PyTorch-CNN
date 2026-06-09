import torch
import pytest
from autovision.model import build_model, BACKBONES


@pytest.mark.parametrize("backbone,n_classes", [
    ("resnet18", 2),
    ("resnet18", 5),
    ("efficientnet_b0", 3),
])
def test_output_shape(backbone, n_classes):
    model = build_model(num_classes=n_classes, backbone=backbone, freeze=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, n_classes), f"Expected (2, {n_classes}), got {out.shape}"


def test_frozen_backbone_keeps_head_trainable():
    model = build_model(num_classes=2, backbone="resnet18", freeze=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen    = [p for p in model.parameters() if not p.requires_grad]
    assert trainable, "classifier head should be trainable"
    assert frozen,    "backbone should be frozen"


def test_unfreeze_makes_everything_trainable():
    model = build_model(num_classes=2, backbone="resnet18", freeze=False)
    frozen = [p for p in model.parameters() if not p.requires_grad]
    assert not frozen, "all params should be trainable when freeze=False"


def test_unknown_backbone_raises():
    with pytest.raises(ValueError, match="backbone must be one of"):
        build_model(num_classes=2, backbone="vgg16")


def test_backbones_constant_matches_supported():
    for b in BACKBONES:
        model = build_model(num_classes=2, backbone=b, freeze=True)
        assert model is not None
