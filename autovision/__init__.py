"""zero-label — train an image classifier from search terms, no dataset required."""

__version__ = "0.2.0"

from autovision.config import Blueprint
from autovision.pipeline import run
from autovision.trainer import predict, train

__all__ = ["Blueprint", "run", "train", "predict", "__version__"]
