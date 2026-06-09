"""AutoVision — zero-dataset image classifier."""

from autovision.config import Blueprint
from autovision.pipeline import run
from autovision.trainer import predict, train

__all__ = ["Blueprint", "run", "train", "predict"]
