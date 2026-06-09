import torch
import pytest
from pathlib import Path
from PIL import Image
from autovision.trainer import image_pipeline


# ── image_pipeline ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("augment", [True, False])
def test_pipeline_output_shape(augment):
    img = Image.new("RGB", (300, 200))
    tensor = image_pipeline(224, augment=augment)(img)
    assert tensor.shape == (3, 224, 224)


def test_pipeline_returns_float32():
    tensor = image_pipeline(128, augment=False)(Image.new("RGB", (100, 100)))
    assert tensor.dtype == torch.float32


def test_pipeline_normalises_values():
    # After ImageNet normalisation white pixels should land around (2.2, 2.4, 2.1)
    white = Image.new("RGB", (64, 64), color=(255, 255, 255))
    t = image_pipeline(64, augment=False)(white)
    assert t.min() > -5.0 and t.max() < 5.0


# ── train + predict (integration) ─────────────────────────────────────────────

def test_train_returns_model_and_classes(tiny_image_dir, tmp_path):
    from autovision.config import Blueprint
    from autovision.trainer import train

    cfg = Blueprint(
        images_dir=tiny_image_dir,
        model_path=str(tmp_path / "model.pt"),
        backbone="resnet18",
        epochs=1,
        batch_size=4,
    )
    model, classes = train(cfg)

    assert set(classes) == {"cat", "dog"}
    assert model is not None


def test_predict_returns_valid_confidences(tiny_image_dir, tmp_path):
    from autovision.config import Blueprint
    from autovision.trainer import train, predict

    model_path = str(tmp_path / "model.pt")
    train(Blueprint(
        images_dir=tiny_image_dir,
        model_path=model_path,
        backbone="resnet18",
        epochs=1,
        batch_size=4,
    ))

    test_img = next(Path(tiny_image_dir).rglob("*.jpg"))
    results = predict(str(test_img), model_path=model_path, top_k=2)

    assert len(results) == 2
    labels = [cls for cls, _ in results]
    confs  = [c   for _, c  in results]

    assert set(labels) <= {"cat", "dog"}
    assert all(0.0 <= c <= 1.0 for c in confs)
    # softmax over both classes must sum to 1
    assert abs(sum(confs) - 1.0) < 1e-4
