"""Training orchestration modules.

Provides trainers for:
- Encoder training (Stage 1): Audio -> Embeddings
- Classifier training (Stage 2): Embeddings -> Ratings
- Joint fine-tune: Unfreeze encoder + classifier, train on audio -> rating
"""

from .encoder_trainer import EncoderTrainer
from .classifier_trainer import ClassifierTrainer
from .joint_finetune_trainer import JointFinetuneTrainer

__all__ = [
    "EncoderTrainer",
    "ClassifierTrainer",
    "JointFinetuneTrainer",
]
