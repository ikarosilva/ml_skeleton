"""User-injectable protocol definitions for music recommendation system."""

from ml_skeleton.protocols.encoder import AudioEncoder
from ml_skeleton.protocols.classifier import RatingClassifier

__all__ = ["AudioEncoder", "RatingClassifier"]
