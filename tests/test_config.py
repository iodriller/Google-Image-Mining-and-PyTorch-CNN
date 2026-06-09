from dataclasses import fields
from autovision.config import Blueprint


def test_defaults_are_sane():
    cfg = Blueprint()
    assert cfg.epochs == 10
    assert cfg.backbone == "efficientnet_b0"
    assert cfg.img_size == 224
    assert cfg.freeze_backbone is True
    assert 0.0 < cfg.val_split < 1.0


def test_custom_values_override_defaults():
    cfg = Blueprint(epochs=3, backbone="resnet18", freeze_backbone=False, lr=5e-4)
    assert cfg.epochs == 3
    assert cfg.backbone == "resnet18"
    assert cfg.freeze_backbone is False
    assert cfg.lr == 5e-4


def test_has_expected_fields():
    names = {f.name for f in fields(Blueprint)}
    assert {"epochs", "backbone", "img_size", "model_path",
            "freeze_backbone", "lr", "batch_size", "val_split"} <= names


def test_two_blueprints_are_independent():
    a = Blueprint(epochs=1)
    b = Blueprint(epochs=99)
    assert a.epochs != b.epochs
