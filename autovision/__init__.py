"""AutoVision — zero-dataset image classifier."""

from autovision.config import TrainConfig
from autovision.pipeline import run
from autovision.trainer import predict, train

__all__ = ["TrainConfig", "run", "train", "predict"]
