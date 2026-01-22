"""
Base tuner abstract class.

Defines the interface that all hyperparameter tuners must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from ml_skeleton.core.protocols import TrainFunction, TrainingContext
from ml_skeleton.core.config import ExperimentConfig
from ml_skeleton.tracking.client import ExplrTracker


class BaseTuner(ABC):
    """
    Abstract base class for hyperparameter tuners.

    All tuner implementations (Optuna, Ray Tune) inherit from this class
    and implement the optimize() method.
    """

    def __init__(
        self,
        train_fn: TrainFunction,
        config: ExperimentConfig,
        mlflow_tracking_uri: str,
    ):
        """
        Initialize the tuner.

        Args:
            train_fn: User's train_model() function
            config: Experiment configuration
            mlflow_tracking_uri: URI of the MLflow tracking server
        """
        self.train_fn = train_fn
        self.config = config
        self.mlflow_tracking_uri = mlflow_tracking_uri

    @abstractmethod
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Returns:
            Dictionary containing:
            - best_params: Best hyperparameters found
            - best_value: Best metric value achieved
            - Additional tuner-specific results
        """
        pass

    @abstractmethod
    def _create_objective(self) -> Callable:
        """Create the objective function for the tuner."""
        pass

    def _build_context(
        self,
        hyperparameters: Dict[str, Any],
        trial_id: Optional[str] = None,
        trial_number: Optional[int] = None,
    ) -> TrainingContext:
        """
        Build TrainingContext for a trial.

        Args:
            hyperparameters: Sampled hyperparameters for this trial
            trial_id: Unique identifier for this trial
            trial_number: Sequential trial number

        Returns:
            Configured TrainingContext
        """
        tracker = ExplrTracker(
            tracking_uri=self.mlflow_tracking_uri,
            experiment_name=self.config.name,
            nested=True,  # Nested runs for tuning trials
        )

        return TrainingContext(
            hyperparameters=hyperparameters,
            tracker=tracker,
            trial_id=trial_id,
            trial_number=trial_number,
            experiment_name=self.config.name,
            seed=self.config.seed,
            checkpoint_dir=self.config.checkpoint_dir,
            artifact_dir=self.config.artifact_dir,
        )
