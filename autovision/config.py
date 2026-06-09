from dataclasses import dataclass


@dataclass
class Blueprint:
    images_dir: str = "images"
    model_path: str = "best_model.pt"
    backbone: str = "efficientnet_b0"
    freeze_backbone: bool = True
    img_size: int = 224
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    val_split: float = 0.15
