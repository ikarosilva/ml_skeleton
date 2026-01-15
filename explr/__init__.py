"""
explr - Deep Learning Training Framework

A framework for training and deploying deep learning models with:
- MLflow experiment tracking
- Hyperparameter tuning (Optuna, Ray Tune)
- Support for PyTorch and TensorFlow
"""

__version__ = "0.1.0"

from explr.core.protocols import TrainingContext, TrainingResult, TrainFunction
from explr.core.config import ExperimentConfig, TuningConfig, MLflowConfig
from explr.runner.experiment import Experiment, run_experiment

__all__ = [
    "TrainingContext",
    "TrainingResult",
    "TrainFunction",
    "ExperimentConfig",
    "TuningConfig",
    "MLflowConfig",
    "Experiment",
    "run_experiment",
]
