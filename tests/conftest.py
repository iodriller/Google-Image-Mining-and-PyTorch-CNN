import pytest
from PIL import Image
from pathlib import Path


@pytest.fixture(scope="session")
def tiny_image_dir(tmp_path_factory):
    """
    Minimal two-class image directory used by integration tests.
    8 synthetic solid-colour images per class — no network, no real photos.
    """
    root = tmp_path_factory.mktemp("images")
    colors = [(200, 80, 60), (60, 120, 200), (180, 200, 60), (90, 60, 180),
              (220, 160, 80), (60, 200, 160), (200, 60, 140), (100, 100, 100)]

    for cls in ("cat", "dog"):
        cls_dir = root / cls
        cls_dir.mkdir()
        for i, color in enumerate(colors):
            Image.new("RGB", (64, 64), color=color).save(cls_dir / f"{i}.jpg")

    return str(root)
