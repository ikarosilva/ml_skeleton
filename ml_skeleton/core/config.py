"""
Configuration dataclasses for the training framework.

Provides structured configuration for experiments, MLflow tracking,
and hyperparameter tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class TunerType(Enum):
    """Available hyperparameter tuning backends."""

    NONE = "none"
    OPTUNA = "optuna"
    RAY_TUNE = "ray_tune"


@dataclass
class MLflowConfig:
    """MLflow server and tracking configuration."""

    # MLflow tracking server URI
    tracking_uri: str = "http://localhost:5000"

    # Default artifact storage location
    artifact_location: str = "./mlruns"

    # Experiment name in MLflow
    experiment_name: str = "default"

    # Model registry URI (optional, for model versioning)
    registry_uri: Optional[str] = None

    # Backend store for MLflow server
    backend_store_uri: str = "sqlite:///mlflow.db"

    # Whether to auto-start MLflow server if not running
    auto_start: bool = True


@dataclass
class SearchSpaceConfig:
    """
    Hyperparameter search space configuration.

    Parameters are defined as a dictionary mapping parameter names
    to their search space definitions.

    Example:
        parameters = {
            "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-1},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
            "hidden_size": {"type": "int", "low": 64, "high": 512, "step": 64},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5}
        }
    """

    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class TuningConfig:
    """Hyperparameter tuning configuration."""

    # Which tuner to use
    tuner_type: TunerType = TunerType.NONE

    # Number of trials to run
    n_trials: int = 100

    # Maximum time for tuning (seconds, None for unlimited)
    timeout: Optional[int] = None

    # Optuna-specific settings
    sampler: str = "TPESampler"  # TPESampler, CmaEsSampler, RandomSampler
    pruner: str = "MedianPruner"  # MedianPruner, HyperbandPruner, NopPruner

    # Ray Tune-specific settings
    scheduler: str = "ASHAScheduler"  # ASHAScheduler, PopulationBasedTraining
    search_alg: str = "OptunaSearch"  # OptunaSearch, HyperOptSearch
    num_samples: int = 100
    max_concurrent_trials: int = 4
    resources_per_trial: Dict[str, Any] = field(
        default_factory=lambda: {"cpu": 4, "gpu": 1}
    )

    # Search space definition
    search_space: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)

    # Optuna storage for persistence (e.g., "sqlite:///optuna.db")
    optuna_storage: Optional[str] = None


@dataclass
class ExperimentConfig:
    """
    Main experiment configuration.

    This is the primary configuration object users create to define
    their experiment setup.

    Example:
        config = ExperimentConfig(
            name="mnist_experiment",
            framework="pytorch",
            hyperparameters={"epochs": 20, "patience": 5},
            tuning=TuningConfig(
                tuner_type=TunerType.OPTUNA,
                n_trials=20
            )
        )
    """

    # Experiment identification
    name: str = "experiment"
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    # Framework selection ("pytorch" or "tensorflow")
    framework: str = "pytorch"

    # Default hyperparameters (used when not tuning or as base values)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # MLflow configuration
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    # Tuning configuration
    tuning: TuningConfig = field(default_factory=TuningConfig)

    # Reproducibility
    seed: Optional[int] = 42
    deterministic: bool = True

    # Paths
    checkpoint_dir: str = "./checkpoints"
    artifact_dir: str = "./artifacts"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from a dictionary, handling nested dataclasses."""
        # Handle nested configs
        if "mlflow" in data and isinstance(data["mlflow"], dict):
            data["mlflow"] = MLflowConfig(**data["mlflow"])

        if "tuning" in data and isinstance(data["tuning"], dict):
            tuning_data = data["tuning"]

            # Convert tuner_type string to enum
            if "tuner_type" in tuning_data:
                tuning_data["tuner_type"] = TunerType(tuning_data["tuner_type"])

            # Handle search space
            if "search_space" in tuning_data and isinstance(
                tuning_data["search_space"], dict
            ):
                tuning_data["search_space"] = SearchSpaceConfig(
                    **tuning_data["search_space"]
                )

            data["tuning"] = TuningConfig(**tuning_data)

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        data = self._to_dict()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary for serialization."""
        data = {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "framework": self.framework,
            "hyperparameters": self.hyperparameters,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "checkpoint_dir": self.checkpoint_dir,
            "artifact_dir": self.artifact_dir,
            "mlflow": {
                "tracking_uri": self.mlflow.tracking_uri,
                "artifact_location": self.mlflow.artifact_location,
                "experiment_name": self.mlflow.experiment_name,
                "backend_store_uri": self.mlflow.backend_store_uri,
                "auto_start": self.mlflow.auto_start,
            },
            "tuning": {
                "tuner_type": self.tuning.tuner_type.value,
                "n_trials": self.tuning.n_trials,
                "timeout": self.tuning.timeout,
                "sampler": self.tuning.sampler,
                "pruner": self.tuning.pruner,
                "scheduler": self.tuning.scheduler,
                "search_alg": self.tuning.search_alg,
                "num_samples": self.tuning.num_samples,
                "max_concurrent_trials": self.tuning.max_concurrent_trials,
                "resources_per_trial": self.tuning.resources_per_trial,
                "search_space": {"parameters": self.tuning.search_space.parameters},
            },
        }
        return data
