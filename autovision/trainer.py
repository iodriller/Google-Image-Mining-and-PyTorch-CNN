from __future__ import annotations

import logging
import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .config import Blueprint
from .model import build_model

logger = logging.getLogger(__name__)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Windows multiprocessing + CUDA can deadlock with workers > 0
_NUM_WORKERS = 0 if os.name == "nt" else 2

# ImageNet normalisation — required by all pretrained torchvision backbones
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def image_pipeline(img_size: int, augment: bool) -> transforms.Compose:
    if augment:
        return transforms.Compose(
            [
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(_MEAN, _STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )


def _partition_data(cfg: Blueprint) -> Tuple[DataLoader, DataLoader, List[str]]:
    # Two ImageFolder instances so train and val get different transforms
    train_ds = ImageFolder(cfg.images_dir, transform=image_pipeline(cfg.img_size, augment=True))
    val_ds = ImageFolder(cfg.images_dir, transform=image_pipeline(cfg.img_size, augment=False))

    n_val = max(1, int(len(train_ds) * cfg.val_split))
    indices = torch.randperm(len(train_ds)).tolist()

    train_loader = DataLoader(
        Subset(train_ds, indices[n_val:]),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(val_ds, indices[:n_val]),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_ds.classes


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None = None,
    desc: str = "",
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if training:
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                out = model(images)
                loss = criterion(out, labels)

        total_loss += loss.item() * images.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train(cfg: Blueprint) -> Tuple[nn.Module, List[str]]:
    """Train a classifier from images already on disk. Returns (best_model, class_names)."""
    print(f"Device: {DEVICE}")

    train_loader, val_loader, class_names = _partition_data(cfg)
    print(f"Classes : {class_names}")
    print(f"Train   : {len(train_loader.dataset)} images | Val: {len(val_loader.dataset)} images")

    model = build_model(len(class_names), cfg.backbone, cfg.freeze_backbone).to(DEVICE)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    # Always persist the first completed epoch. A tiny or difficult validation
    # split can score 0.0, which is still a valid trained checkpoint.
    best_val_acc = -1.0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer,
            desc=f"Epoch {epoch}/{cfg.epochs} [train]",
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion,
            desc=f"Epoch {epoch}/{cfg.epochs} [val]",
        )
        scheduler.step()

        marker = " *" if val_acc > best_val_acc else ""
        print(
            f"Epoch {epoch:2d}/{cfg.epochs} | "
            f"train  loss {train_loss:.4f}  acc {train_acc:.3f} | "
            f"val  loss {val_loss:.4f}  acc {val_acc:.3f}{marker}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"state_dict": model.state_dict(), "classes": class_names, "config": cfg},
                cfg.model_path,
            )

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {cfg.model_path}")
    return model, class_names


def predict(
    image_path: str,
    model_path: str = "best_model.pt",
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """Load a saved checkpoint and return top-k (class, confidence) pairs."""
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    class_names: List[str] = checkpoint["classes"]
    cfg: Blueprint = checkpoint["config"]

    model = build_model(len(class_names), cfg.backbone, freeze=False).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    tensor = image_pipeline(cfg.img_size, augment=False)(
        PILImage.open(image_path).convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    k = min(top_k, len(class_names))
    topk_probs, topk_idx = probs.topk(k)
    return [(class_names[i], float(p)) for i, p in zip(topk_idx.tolist(), topk_probs.tolist())]
