from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model import build_model

logger = logging.getLogger(__name__)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# ImageNet normalisation — required by all pretrained torchvision backbones
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


@dataclass
class TrainConfig:
    images_dir: str = "images"
    model_path: str = "best_model.pt"
    backbone: str = "efficientnet_b0"
    freeze_backbone: bool = True
    img_size: int = 224
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    val_split: float = 0.15


def _make_transforms(img_size: int, augment: bool) -> transforms.Compose:
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


def _build_loaders(
    cfg: TrainConfig,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    # Two ImageFolder instances so train and val get different transforms
    train_ds = ImageFolder(cfg.images_dir, transform=_make_transforms(cfg.img_size, augment=True))
    val_ds = ImageFolder(cfg.images_dir, transform=_make_transforms(cfg.img_size, augment=False))

    n_val = max(1, int(len(train_ds) * cfg.val_split))
    indices = torch.randperm(len(train_ds)).tolist()

    train_loader = DataLoader(
        Subset(train_ds, indices[n_val:]),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, indices[:n_val]),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
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


def train(cfg: TrainConfig) -> Tuple[nn.Module, List[str]]:
    """Scrape → split → train → checkpoint. Returns (best_model, class_names)."""
    print(f"Device: {DEVICE}")

    train_loader, val_loader, class_names = _build_loaders(cfg)
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    print(f"Classes : {class_names}")
    print(f"Train   : {n_train} images | Val: {n_val} images")

    model = build_model(len(class_names), cfg.backbone, cfg.freeze_backbone).to(DEVICE)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

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
                {
                    "state_dict": model.state_dict(),
                    "classes": class_names,
                    "config": cfg,
                },
                cfg.model_path,
            )

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {cfg.model_path}")
    return model, class_names
