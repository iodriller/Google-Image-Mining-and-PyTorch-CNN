"""End-to-end pipeline: scrape images → train a custom classifier.

Usage
-----
    from autovision import run

    model, classes = run(
        queries=["golden retriever", "siberian husky", "german shepherd"],
        n_images=150,
        epochs=10,
    )
"""

from __future__ import annotations

from typing import List, Tuple

import torch.nn as nn

from .config import TrainConfig
from .scraper import ImageScraper
from .trainer import train


def run(
    queries: List[str],
    n_images: int = 100,
    images_dir: str = "images",
    epochs: int = 10,
    batch_size: int = 32,
    backbone: str = "efficientnet_b0",
    freeze_backbone: bool = True,
) -> Tuple[nn.Module, List[str]]:
    """Download images for each query, then train and return the best model."""
    scraper = ImageScraper(images_dir=images_dir)
    for query in queries:
        scraper.search_and_download(query, n_images=n_images)

    cfg = TrainConfig(
        images_dir=images_dir,
        epochs=epochs,
        batch_size=batch_size,
        backbone=backbone,
        freeze_backbone=freeze_backbone,
    )
    return train(cfg)
