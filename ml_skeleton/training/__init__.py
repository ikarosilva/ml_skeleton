"""Training orchestration modules.

Provides trainers for:
- Encoder training (Stage 1): Audio -> Embeddings
- Classifier training (Stage 2): Embeddings -> Ratings
"""

from .encoder_trainer import EncoderTrainer
from .classifier_trainer import ClassifierTrainer

__all__ = [
    "EncoderTrainer",
    "ClassifierTrainer"
]
